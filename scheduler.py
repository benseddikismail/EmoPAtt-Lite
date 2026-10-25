"""Per-step cosine learning-rate schedule with a linear warmup."""

import math


class CosineWithWarmup:
    """Scales every param group's base LR by a warmup-then-cosine factor.

    Stepping per iteration rather than per epoch matters here: FER2013 is small
    enough that an epoch is only ~220 steps, and a per-epoch schedule makes the
    decay far too coarse.
    """

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = min(warmup_steps, self.total_steps - 1)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self.step(0)

    def factor(self, step):
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine

    def step(self, step=None):
        if step is None:
            step = self.step_count
            self.step_count += 1
        f = self.factor(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * f
        return f

    @property
    def last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]
