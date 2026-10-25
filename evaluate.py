"""Evaluate one or more trained EmoPAtt-Lite checkpoints on FER2013.
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from data import CLASSES, load_split
from model import build_model
from train import evaluate, pick_device


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    model = build_model(args, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), args


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main(args):
    device = pick_device(args.device)
    split_x, split_y = load_split(args.split, device)
    y_true = split_y.cpu().numpy()

    total = None
    for path in args.checkpoints:
        model, ckpt_args = load_checkpoint(path, device)
        acc, probs = evaluate(
            model,
            split_x,
            split_y,
            args.batch_size,
            device,
            tta=not args.no_tta,
            return_logits=True,
        )
        print(f"{path}  ({ckpt_args.backbone} @ {ckpt_args.img_size})  {args.split} acc {acc:.4f}")
        total = probs if total is None else total + probs
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    probs = (total / len(args.checkpoints)).numpy()
    pred = probs.argmax(axis=1)
    acc = float((pred == y_true).mean())
    print(f"\nensemble of {len(args.checkpoints)}: {args.split} accuracy {acc:.4f}")

    cm = confusion(y_true, pred, len(CLASSES))
    width = max(len(c) for c in CLASSES)
    print("\nconfusion matrix (rows = truth)")
    print(" " * (width + 2) + "  ".join(f"{c[:4]:>5s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>{width}s}  " + "  ".join(f"{v:5d}" for v in cm[i]))

    print("\nper-class precision / recall / f1")
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {cls:>{width}s}  {prec:.3f}  {rec:.3f}  {f1:.3f}   (n={cm[i].sum()})")

    if args.save:
        np.savez(args.save, probs=probs, labels=y_true)
        print(f"\nsaved probabilities to {args.save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="paths to runs/<name>/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--save", default=None)
    main(p.parse_args())
