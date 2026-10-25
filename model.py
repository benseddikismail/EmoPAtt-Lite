"""EmoPAtt-Lite: a truncated lightweight backbone plus an attention classifier."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from elements import AttentionClassifier, CBAM, PatchExtraction, SpatialTransformer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EmoPAttLite(nn.Module):
    """Pipeline: STN -> truncated ImageNet backbone -> patch extraction -> attention head.

    The network is fed the raw 48x48 FER2013 crop in [0, 1]; upsampling to the
    backbone's native resolution happens on the GPU inside ``forward`` so the
    dataloader only ever moves 48x48 tensors around.
    """

    def __init__(
        self,
        backbone="mobilenetv1_100",
        pretrained=True,
        num_classes=7,
        img_size=224,
        embed_dim=256,
        num_heads=4,
        dropout=0.1,
        patch_stride=2,
        stage=3,
        use_stn=True,
        use_cbam=True,
        drop_path=0.05,
    ):
        super().__init__()
        self.img_size = img_size

        kwargs = dict(pretrained=pretrained, features_only=True, out_indices=(stage,))
        try:
            self.backbone = timm.create_model(backbone, drop_path_rate=drop_path, **kwargs)
        except TypeError:  # backbone does not support stochastic depth
            self.backbone = timm.create_model(backbone, **kwargs)
        feat_channels = self.backbone.feature_info.channels()[-1]

        self.stn = SpatialTransformer(in_channels=3) if use_stn else None
        self.cbam = CBAM(feat_channels) if use_cbam else nn.Identity()
        self.patch_extraction = PatchExtraction(feat_channels, embed_dim, stride=patch_stride)
        self.classifier = AttentionClassifier(
            dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bicubic",
                align_corners=False,
            ).clamp_(0, 1)
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.preprocess(x)
        if self.stn is not None:
            x = self.stn(x)
        x = self.backbone(x)[-1]
        x = self.cbam(x)
        x = self.patch_extraction(x)
        return self.classifier(x)

    def param_groups(self, head_lr, backbone_lr, weight_decay):
        """Split parameters so the pretrained trunk gets a smaller learning rate
        than the freshly initialised head, and so norm/bias terms skip decay."""
        groups = {
            "backbone_decay": dict(params=[], lr=backbone_lr, weight_decay=weight_decay),
            "backbone_nodecay": dict(params=[], lr=backbone_lr, weight_decay=0.0),
            "head_decay": dict(params=[], lr=head_lr, weight_decay=weight_decay),
            "head_nodecay": dict(params=[], lr=head_lr, weight_decay=0.0),
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            where = "backbone" if name.startswith("backbone.") else "head"
            decay = "nodecay" if param.ndim <= 1 or name.endswith("pos_embed") else "decay"
            groups[f"{where}_{decay}"]["params"].append(param)
        return [g for g in groups.values() if g["params"]]

    def freeze_backbone(self, frozen=True):
        for param in self.backbone.parameters():
            param.requires_grad_(not frozen)
        self.backbone.train(not frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Keep a frozen trunk in eval mode so its BatchNorm statistics stay put.
        if mode and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self


def build_model(args, num_classes=7):
    return EmoPAttLite(
        backbone=args.backbone,
        pretrained=True,
        num_classes=num_classes,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        patch_stride=args.patch_stride,
        stage=args.stage,
        use_stn=not args.no_stn,
        use_cbam=not args.no_cbam,
        drop_path=args.drop_path,
    )
