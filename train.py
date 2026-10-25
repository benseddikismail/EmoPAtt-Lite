"""Train EmoPAtt-Lite on FER2013.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CLASSES, Batches, GPUAugment, load_split, mixup_cutmix, one_hot, tta_views
from elements import ModelEMA
from model import build_model
from scheduler import CosineWithWarmup


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_ctx(device, mode):
    if mode == "off" or device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def soft_cross_entropy(logits, target):
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


@torch.no_grad()
def evaluate(model, x, y, batch_size, device, amp="off", tta=False, return_logits=False):
    model.eval()
    probs = []
    for xb, _ in Batches(x, y, batch_size):
        views = tta_views(xb) if tta else [xb]
        acc = None
        for view in views:
            with autocast_ctx(device, amp):
                out = F.softmax(model(view).float(), dim=-1)
            acc = out if acc is None else acc + out
        probs.append((acc / len(views)).cpu())
    probs = torch.cat(probs)
    pred = probs.argmax(dim=1)
    accuracy = (pred == y.cpu()).float().mean().item()
    return (accuracy, probs) if return_logits else accuracy


def save_best(state, args, out_dir):
    """Persist the best weights seen so far.

    Written every time validation improves rather than once at the end, so an
    interrupted run still leaves a usable checkpoint behind.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        dict(
            state_dict=state["best_state"],
            args=vars(args),
            best_val=state["best_val"],
            epoch=state["epoch"],
            is_ema=state["best_is_ema"],
        ),
        os.path.join(out_dir, "best.pt"),
    )
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(dict(history=state["history"], args=vars(args), best_val=state["best_val"]), fh, indent=2)


def run_stage(
    name,
    model,
    ema,
    opt,
    sched,
    train_x,
    train_y,
    val_x,
    val_y,
    augment,
    args,
    device,
    epochs,
    state,
    out_dir,
):
    num_classes = len(CLASSES)
    for epoch in range(epochs):
        model.train()
        started = time.time()
        total_loss, total_correct, total_seen = 0.0, 0, 0

        for xb, yb in Batches(train_x, train_y, args.batch_size, shuffle=True, drop_last=True):
            xb = augment(xb)
            if args.mixup > 0 or args.cutmix > 0:
                xb, target = mixup_cutmix(
                    xb,
                    yb,
                    num_classes,
                    mixup_alpha=args.mixup,
                    cutmix_alpha=args.cutmix,
                    prob=args.mix_prob,
                    smoothing=args.smoothing,
                )
            else:
                target = one_hot(yb, num_classes, args.smoothing)

            with autocast_ctx(device, args.amp):
                logits = model(xb)
                loss = soft_cross_entropy(logits.float(), target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += yb.size(0)

        val_acc = evaluate(model, val_x, val_y, args.eval_batch_size, device, args.amp)
        ema_acc = (
            evaluate(ema.module, val_x, val_y, args.eval_batch_size, device, args.amp)
            if ema is not None
            else 0.0
        )
        best_of_two = max(val_acc, ema_acc)

        state["epoch"] += 1
        tag = ""
        if best_of_two > state["best_val"]:
            state["best_val"] = best_of_two
            source = ema.module if ema_acc >= val_acc and ema is not None else model
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in source.state_dict().items()}
            state["best_is_ema"] = bool(ema is not None and ema_acc >= val_acc)
            tag = "  *"

        print(
            f"[{name}] epoch {state['epoch']:3d}/{state['total_epochs']}  "
            f"loss {total_loss / total_seen:.4f}  train {total_correct / total_seen:.4f}  "
            f"val {val_acc:.4f}  ema {ema_acc:.4f}  "
            f"lr {sched.last_lr[0]:.2e}  {time.time() - started:.0f}s{tag}",
            flush=True,
        )
        state["history"].append(
            dict(
                stage=name,
                epoch=state["epoch"],
                loss=total_loss / total_seen,
                train_acc=total_correct / total_seen,
                val_acc=val_acc,
                ema_acc=ema_acc,
            )
        )
        if tag:
            save_best(state, args, out_dir)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}  backbone: {args.backbone}  img_size: {args.img_size}")

    train_x, train_y = load_split("train", device)
    val_x, val_y = load_split("val", device)
    test_x, test_y = load_split("test", device)
    print(f"train {len(train_x)}  val {len(val_x)}  test {len(test_x)}")

    model = build_model(args, num_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")

    augment = GPUAugment(
        degrees=args.rotate,
        translate=args.translate,
        scale=(args.scale_min, args.scale_max),
        brightness=args.brightness,
        contrast=args.contrast,
        erase_prob=args.erase,
    )

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    steps_per_epoch = len(train_x) // args.batch_size
    state = dict(
        epoch=0,
        total_epochs=args.train_epochs + args.ft_epochs,
        best_val=0.0,
        best_state=None,
        best_is_ema=False,
        history=[],
    )

    # Stage 1 -- frozen trunk, head only.
    if args.train_epochs > 0:
        model.freeze_backbone(True)
        opt = torch.optim.AdamW(
            model.param_groups(args.learning_rate, args.learning_rate, args.weight_decay)
        )
        sched = CosineWithWarmup(
            opt,
            total_steps=args.train_epochs * steps_per_epoch,
            warmup_steps=min(steps_per_epoch, 200),
        )
        run_stage(
            "head", model, None, opt, sched, train_x, train_y, val_x, val_y,
            augment, args, device, args.train_epochs, state, out_dir,
        )

    # Stage 2 -- end-to-end fine-tune.
    model.freeze_backbone(False)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(
        model.param_groups(args.ft_head_lr, args.ft_lr, args.weight_decay)
    )
    sched = CosineWithWarmup(
        opt,
        total_steps=args.ft_epochs * steps_per_epoch,
        warmup_steps=min(2 * steps_per_epoch, 500),
        min_lr_ratio=args.min_lr_ratio,
    )
    run_stage(
        "finetune", model, ema, opt, sched, train_x, train_y, val_x, val_y,
        augment, args, device, args.ft_epochs, state, out_dir,
    )

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    source = "EMA" if state["best_is_ema"] else "raw"
    print(f"\nbest val accuracy {state['best_val']:.4f} ({source} weights)")

    val_acc, val_probs = evaluate(
        model, val_x, val_y, args.eval_batch_size, device, args.amp, tta=True, return_logits=True
    )
    test_acc, test_probs = evaluate(
        model, test_x, test_y, args.eval_batch_size, device, args.amp, return_logits=True
    )
    test_tta, test_probs_tta = evaluate(
        model, test_x, test_y, args.eval_batch_size, device, args.amp, tta=True, return_logits=True
    )
    print(f"val  (TTA): {val_acc:.4f}")
    print(f"test       : {test_acc:.4f}")
    print(f"test (TTA) : {test_tta:.4f}")

    torch.save(
        dict(
            state_dict=model.state_dict(),
            args=vars(args),
            best_val=state["best_val"],
            test_acc=test_acc,
            test_acc_tta=test_tta,
        ),
        os.path.join(out_dir, "best.pt"),
    )
    np.savez(
        os.path.join(out_dir, "probs.npz"),
        val=val_probs.numpy(),
        test=test_probs.numpy(),
        test_tta=test_probs_tta.numpy(),
    )
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(dict(history=state["history"], args=vars(args), test_acc=test_acc,
                       test_acc_tta=test_tta, best_val=state["best_val"]), fh, indent=2)
    print(f"saved to {out_dir}")

    per_class = {}
    pred = test_probs_tta.argmax(dim=1)
    y_true = test_y.cpu()
    for i, cls in enumerate(CLASSES):
        mask = y_true == i
        per_class[cls] = (pred[mask] == i).float().mean().item()
    print("per-class recall (TTA):", {k: round(v, 3) for k, v in per_class.items()})


def build_parser():
    p = argparse.ArgumentParser(description="Train EmoPAtt-Lite on FER2013")
    p.add_argument("--name", type=str, default="emopatt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--train_epochs", type=int, default=4, help="frozen-trunk head warmup")
    p.add_argument("--ft_epochs", type=int, default=296, help="end-to-end fine-tune")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3, help="stage-1 LR")
    p.add_argument("--ft_lr", type=float, default=2e-4, help="stage-2 backbone LR")
    p.add_argument("--ft_head_lr", type=float, default=1e-3, help="stage-2 head LR")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")

    p.add_argument("--backbone", type=str, default="mobilenetv1_100")
    p.add_argument("--stage", type=int, default=3, help="backbone feature stage (3 = stride 16)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_stride", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--drop_path", type=float, default=0.05)
    p.add_argument("--no_stn", action="store_true")
    p.add_argument("--no_cbam", action="store_true")

    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mix_prob", type=float, default=0.5)
    p.add_argument("--rotate", type=float, default=15.0)
    p.add_argument("--translate", type=float, default=0.12)
    p.add_argument("--scale_min", type=float, default=0.85)
    p.add_argument("--scale_max", type=float, default=1.18)
    p.add_argument("--brightness", type=float, default=0.25)
    p.add_argument("--contrast", type=float, default=0.25)
    p.add_argument("--erase", type=float, default=0.3)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
