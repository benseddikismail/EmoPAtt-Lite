"""FER2013 loading and GPU-side augmentation.

The whole dataset is 35k 48x48 grayscale images (~82 MB as uint8), so it is held
resident on the accelerator and augmented in batch. That removes the dataloader
from the critical path entirely, which matters a lot on MPS where host/device
round trips dominate for inputs this small.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NPZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "fer2013.npz")


def load_split(split, device, path=NPZ_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python dataset/load_dataset.py` first."
        )
    with np.load(path, allow_pickle=True) as data:
        x = torch.from_numpy(data[f"{split}_x"]).unsqueeze(1).to(device)
        y = torch.from_numpy(data[f"{split}_y"]).to(device)
    return x, y


class Batches:
    """Index-shuffling iterator over a device-resident tensor pair."""

    def __init__(self, x, y, batch_size, shuffle=False, drop_last=False, generator=None):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __len__(self):
        n = len(self.x)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.x)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator, device=self.x.device)
        else:
            order = torch.arange(n, device=self.x.device)
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield self.x[idx].float().div_(255.0), self.y[idx]


def _affine_theta(batch, degrees, translate, scale, device, generator):
    angle = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * math.radians(degrees)
    tx = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * translate
    ty = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * translate
    zoom = torch.empty(batch, device=device).uniform_(scale[0], scale[1], generator=generator)

    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    return theta


class GPUAugment:
    """Flip, affine jitter, photometric jitter and random erasing, all batched."""

    def __init__(
        self,
        degrees=15.0,
        translate=0.12,
        scale=(0.85, 1.18),
        brightness=0.25,
        contrast=0.25,
        erase_prob=0.3,
        erase_scale=(0.02, 0.25),
        generator=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale
        self.generator = generator

    def __call__(self, x):
        b, _, h, w = x.shape
        device = x.device
        g = self.generator

        flip = torch.rand(b, 1, 1, 1, device=device, generator=g) < 0.5
        x = torch.where(flip, x.flip(-1), x)

        theta = _affine_theta(b, self.degrees, self.translate, self.scale, device, g)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")

        if self.brightness or self.contrast:
            bright = 1 + (torch.rand(b, 1, 1, 1, device=device, generator=g) * 2 - 1) * self.brightness
            cont = 1 + (torch.rand(b, 1, 1, 1, device=device, generator=g) * 2 - 1) * self.contrast
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - mean) * cont + mean) * bright
            x = x.clamp_(0, 1)

        if self.erase_prob > 0:
            x = self._erase(x, h, w, device, g)
        return x

    def _erase(self, x, h, w, device, g):
        b = x.shape[0]
        area = h * w * torch.empty(b, device=device).uniform_(*self.erase_scale, generator=g)
        aspect = torch.exp(
            torch.empty(b, device=device).uniform_(math.log(0.3), math.log(3.3), generator=g)
        )
        eh = (torch.sqrt(area * aspect)).clamp(1, h - 1).round()
        ew = (torch.sqrt(area / aspect)).clamp(1, w - 1).round()
        y0 = (torch.rand(b, device=device, generator=g) * (h - eh)).floor()
        x0 = (torch.rand(b, device=device, generator=g) * (w - ew)).floor()

        ys = torch.arange(h, device=device).view(1, 1, h, 1)
        xs = torch.arange(w, device=device).view(1, 1, 1, w)
        y0, eh = y0.view(b, 1, 1, 1), eh.view(b, 1, 1, 1)
        x0, ew = x0.view(b, 1, 1, 1), ew.view(b, 1, 1, 1)
        box = (ys >= y0) & (ys < y0 + eh) & (xs >= x0) & (xs < x0 + ew)
        active = (torch.rand(b, 1, 1, 1, device=device, generator=g) < self.erase_prob)
        noise = torch.rand_like(x)
        return torch.where(box & active, noise, x)


def one_hot(y, num_classes, smoothing=0.0):
    off = smoothing / num_classes
    target = torch.full((y.size(0), num_classes), off, device=y.device)
    return target.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + off)


def mixup_cutmix(x, y, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, smoothing=0.1):
    """Return soft targets, applying mixup or cutmix to the batch with `prob`."""
    target = one_hot(y, num_classes, smoothing)
    if prob <= 0 or torch.rand(()) >= prob:
        return x, target

    perm = torch.randperm(x.size(0), device=x.device)
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(()) < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))

    if use_cutmix:
        h, w = x.shape[-2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - ((y1 - y0) * (x1 - x0) / (h * w))
    else:
        x = x * lam + x[perm] * (1 - lam)

    return x, target * lam + target[perm] * (1 - lam)


def tta_views(x, scales=(1.0, 0.9)):
    """Deterministic test-time views: each scale, with and without a flip."""
    views = []
    for s in scales:
        if s == 1.0:
            base = x
        else:
            theta = torch.zeros(x.size(0), 2, 3, device=x.device)
            theta[:, 0, 0] = theta[:, 1, 1] = s
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            base = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
        views.append(base)
        views.append(base.flip(-1))
    return views
