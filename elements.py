"""Building blocks for EmoPAtt-Lite."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation over channels."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        hidden = max(channels // reduction_ratio, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=True)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel attention then spatial).

    The channel branch shares one MLP between the average- and max-pooled
    descriptors and sums them *before* the sigmoid; the spatial branch pools
    across channels, not across space.
    """

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        x = x * torch.sigmoid(avg + mx)

        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))


class SpatialTransformer(nn.Module):
    """Affine STN that learns to re-crop the face before feature extraction.

    The localisation head runs on a small copy of the input so it costs almost
    nothing, and it is zero-initialised so the module starts as the identity.
    The predicted delta is bounded with a tanh, which keeps the sampling grid
    from collapsing early in training -- an unconstrained STN very easily folds
    the image onto a single pixel and never recovers.
    """

    def __init__(self, in_channels=3, loc_size=48, max_translate=0.2, max_delta=0.25):
        super().__init__()
        self.loc_size = loc_size
        self.max_translate = max_translate
        self.max_delta = max_delta
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.theta = nn.Linear(32, 6)
        nn.init.zeros_(self.theta.weight)
        nn.init.zeros_(self.theta.bias)

    def forward(self, x):
        small = F.interpolate(
            x, size=(self.loc_size, self.loc_size), mode="bilinear", align_corners=False
        )
        delta = torch.tanh(self.theta(self.localization(small))).view(-1, 2, 3)
        scale = delta.new_tensor(
            [
                [self.max_delta, self.max_delta, self.max_translate],
                [self.max_delta, self.max_delta, self.max_translate],
            ]
        )
        identity = delta.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        theta = identity + delta * scale

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


class SeparableConv(nn.Module):
    """Depthwise 3x3 (or kxk) followed by a pointwise 1x1, each with BN+ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        return F.relu(self.bn2(self.pointwise(x)), inplace=True)


class PatchExtraction(nn.Module):
    """Re-encode the truncated backbone's feature map into a grid of patch tokens.

    Unlike the original, the stride-4 / stride-2 pair is replaced by a single
    stride-2 stage: collapsing a 14x14 map down to 2x2 leaves the attention
    classifier with four tokens, which is too few for attention to do anything.
    """

    def __init__(self, in_channels, out_channels=256, stride=2, use_se=True):
        super().__init__()
        self.conv1 = SeparableConv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = SeparableConv(out_channels, out_channels, 3, stride=1, padding=1)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.project(x)
        return self.se(x)


class AttentionClassifier(nn.Module):
    """Self-attention over the patch tokens, then attention pooling + a linear head.

    The original applied Keras' Attention layer to a length-1 sequence, where
    softmax over a single key returns the value unchanged -- an exact no-op.
    Here attention runs over the HxW patch tokens, so it can actually weight
    facial regions against each other.
    """

    def __init__(
        self,
        dim=256,
        num_heads=4,
        num_classes=7,
        mlp_ratio=2.0,
        dropout=0.1,
        max_tokens=256,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.pool_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, C)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : h * w]

        y = self.norm1(tokens)
        tokens = tokens + self.attn(y, y, y, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))

        q = self.pool_query.expand(b, -1, -1)
        pooled = self.pool_attn(q, tokens, tokens, need_weights=False)[0].squeeze(1)
        pooled = self.norm3(pooled + tokens.mean(dim=1))

        return self.head(self.dropout(pooled))


class ModelEMA(nn.Module):
    """Exponential moving average of the weights, evaluated alongside the model.

    Updates run through the fused ``_foreach`` ops: a per-tensor loop costs two
    kernel launches per parameter per step, which is a measurable slice of the
    step time on MPS.
    """

    def __init__(self, model, decay=0.9998, warmup=2000):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

        ema_state, model_state = self.module.state_dict(), model.state_dict()
        self._float_keys = [k for k, v in ema_state.items() if v.dtype.is_floating_point]
        self._other_keys = [k for k in ema_state if k not in set(self._float_keys)]
        self._ema_floats = [ema_state[k] for k in self._float_keys]
        self._src_floats = [model_state[k] for k in self._float_keys]
        self._ema_other = [ema_state[k] for k in self._other_keys]
        self._src_other = [model_state[k] for k in self._other_keys]

    @torch.no_grad()
    def update(self, model=None):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        torch._foreach_mul_(self._ema_floats, d)
        torch._foreach_add_(self._ema_floats, self._src_floats, alpha=1 - d)
        for ema_v, src_v in zip(self._ema_other, self._src_other):
            ema_v.copy_(src_v)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
