import math
from tensorflow.keras import backend as K

class CosineAnnealingWithWarmRestartsLR(tf.keras.callbacks.Callback):
    def __init__(
        self,
        optimizer,
        warmup_steps=128,
        cycle_steps=512,
        min_lr=0.0,
        max_lr=1e-3
    ):
        super().__init__()
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_counter = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.steps_counter += 1
        current_cycle_steps = self.steps_counter % self.cycle_steps
        if current_cycle_steps < self.warmup_steps:
            current_lr = (
                self.min_lr
                + (self.max_lr - self.min_lr) * current_cycle_steps / self.warmup_steps
            )
        else:
            current_lr = (
                self.min_lr
                + (self.max_lr - self.min_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (current_cycle_steps - self.warmup_steps)
                        / (self.cycle_steps - self.warmup_steps)
                    )
                )
                / 2
            )
        K.set_value(self.optimizer.lr, current_lr)