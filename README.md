# EmoPAtt-Lite: Lightweight Facial Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://doi.org/10.1007/978-3-032-21174-3_10)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--21174--3__10-orange)](https://doi.org/10.1007/978-3-032-21174-3_10)

Reference implementation for:

> **EmoPAtt-Lite: Lightweight Facial Emotion Recognition**
> Ismail Ben Seddik, Adebowale Emmanuel Adelekan
> *Proceedings of the International Conference on Information Technology and Applications (ICITA)*,
> Lecture Notes in Networks and Systems, Springer Nature Switzerland, 2026, pp. 102–115.
> [doi:10.1007/978-3-032-21174-3_10](https://doi.org/10.1007/978-3-032-21174-3_10)

## Abstract

Facial expressions are a fundamental component of human communication, conveying a wide range of
emotions. However, automatic facial expression recognition (FER) in the wild remains challenging,
particularly under adverse conditions. Recent advances in computer vision have demonstrated the
effectiveness of deep neural networks for FER, but their deployment is often constrained by the need
for substantial computational resources. To address this, we propose EmoPAtt-Lite, a compact FER
model that modifies MobileNetV1 by integrating spatial adaptation and channel-aware recalibration
modules. Unlike prior patch-attention methods, our model emphasizes spatial alignment (via a Spatial
Transformer Network) and channel weighting (via Squeeze-and-Excitation block) to enhance lightweight
FER performance. Despite its compressed size of only 1.3M parameters, EmoPAtt-Lite achieves
state-of-the-art performance on the FER2013 benchmark, reaching an accuracy of 79.35%, thus
demonstrating that high recognition accuracy can be attained without heavy computational demands.

## Architecture

```
48x48 grayscale  ──►  bicubic upsample to 224x224, replicate to 3 channels, ImageNet normalise
                 ──►  Spatial Transformer Network      (affine, identity-initialised, bounded)
                 ──►  MobileNetV1 truncated at stride 16   (ImageNet pre-trained, 512 channels)
                 ──►  CBAM                             (channel then spatial recalibration)
                 ──►  Patch extraction                 (SeparableConv x2 ─► 1x1 conv ─► SE block)
                 ──►  Attention classifier             (MHSA over 7x7 tokens ─► attention pool)
                 ──►  Linear ─► 7 emotions
```

Three design points matter for accuracy:

- **The STN is bounded and identity-initialised.** An unconstrained affine STN folds the sampling
  grid onto a few pixels within the first epochs and never recovers; the predicted transform here is
  a `tanh`-bounded delta around the identity.
- **Patch extraction keeps a 7x7 token grid.** Collapsing the 14x14 backbone map all the way to 2x2
  leaves the attention stage with four tokens, which is too few for attention to express anything.
- **Attention runs over the patch tokens, not over a length-1 sequence.** Softmax over a single key
  returns its value unchanged, so an attention layer applied to one pooled vector is an exact
  identity function.

Upsampling happens on the accelerator inside `forward`, so the input pipeline only ever moves 48x48
tensors; the whole dataset stays resident on the GPU and augmentation is applied per batch.

## Results

FER2013 uses the standard split: 28,709 training images, 3,589 `PublicTest` for validation and
model selection, and 3,589 `PrivateTest` for the reported test accuracy.

| Run | Params | Input | Epochs | FER2013 test |
| --- | --- | --- | --- | --- |
| EmoPAtt-Lite — figure reported in the paper | 1.3M | — | — | **79.35%** |
| This repository, default configuration | 4.41M | 224x224 | 300 | not yet run to completion |
| This repository, `--img_size 112 --ft_epochs 27` | 4.41M | 112x112 | 30 | 70.97% (69.13% val) |

The 112x112 row is a measured, reproducible result from a complete run, including test-time
augmentation; `runs/` is written by the commands below. The default 300-epoch configuration is the
full-length recipe the short run extrapolates to — a 120-epoch run at 224x224 was interrupted at
epoch 70 with validation accuracy at 71.4% and still climbing, so the default schedule has not been
measured end to end in this repository and the table does not claim a number for it.

For context on what the FER2013 private test set supports: published single models sit at roughly
73-74% (VGG 73.28%, ResMaskingNet 74.14%, LHC-Net 74.42%) and the entries above 75% are ensembles
(Pramerdorfer et al. 75.2%, ResMaskingNet ensemble 76.82%). Reaching the paper's figure is expected
to need the full schedule, and `train_ensemble.sh` is provided for the soft-vote step.

Parameter breakdown of this implementation: backbone 3.21M, attention classifier 0.86M, patch
extraction 0.28M, CBAM 0.03M, STN 0.03M. This is larger than the 1.3M quoted in the abstract: the
backbone here is truncated at stride 16, keeping MobileNetV1's full 512-channel stage.

## Installation

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.14 on Apple Silicon (MPS). CUDA and CPU both work; the device is selected
automatically and can be overridden with `--device`.

## Dataset

`dataset/fer2013.tar.gz` ships with the repository. Unpack it into a single lossless archive:

```bash
python dataset/load_dataset.py
```

This writes `dataset/fer2013.npz` holding the raw 48x48 pixels as `uint8`. The pixels are kept
lossless on purpose: the earlier pipeline re-encoded every face as a JPEG, which adds compression
artefacts to an image only 48 pixels across and makes each epoch pay for 35k JPEG decodes. Pass
`--images` to also export the `ImageFolder` layout, as lossless PNGs.

## Training

```bash
python train.py
```

The defaults are the full-length configuration. Written out in full:

```bash
python train.py --name emopatt --backbone mobilenetv1_100 --img_size 224 \
                --train_epochs 4 --ft_epochs 296 --batch_size 128 --seed 0
```

The schedule has two stages:

| Stage | Epochs | Trainable | LR |
| --- | --- | --- | --- |
| Head warmup | 4 | classifier only, trunk frozen in eval mode | 1e-3 |
| Fine-tune | 296 | end to end | trunk 2e-4, head 1e-3 |

The schedule is long on purpose. Mixup and CutMix suppress training accuracy for most of the run —
in the short runs above, training accuracy stayed *below* validation accuracy throughout, which is
the signature of a model held back by regularisation rather than one that has converged. Shortening
the schedule mostly removes the low-learning-rate phase at the end, which is where the cosine
decay delivers the largest single jump (+3.9 points over the final third of the 30-epoch run).

For a quick iteration cycle, `--img_size 112 --train_epochs 3 --ft_epochs 27` finishes in about
15 minutes on an M1 Max and reaches ~71%.

AdamW with weight decay 0.05 (norm and bias terms excluded), per-step cosine decay with linear
warmup, gradient clipping at 3.0, and an EMA copy of the weights (decay 0.9998) evaluated alongside
the raw model each epoch. The checkpoint kept is whichever of the two scores higher on validation.

Regularisation: label smoothing 0.1, mixup (alpha 0.2) and CutMix (alpha 1.0) applied to 50% of
batches, horizontal flip, affine jitter (+/-15 degrees, +/-12% translation, 0.85-1.18 scale),
brightness and contrast jitter (+/-25%), and random erasing (p=0.3).

Training writes to `runs/<name>/`: `best.pt` (rewritten whenever validation improves, so an
interrupted run still leaves a usable checkpoint), `history.json`, and `probs.npz` with the
per-split softmax outputs.

On an M1 Max a 224x224 epoch takes ~100s, so the default 300-epoch schedule is roughly 8 hours.
Accuracy is unchanged by `--amp`; fp16 and bf16 autocast both run *slower* than fp32 on MPS, and
`channels_last` is far slower still, so both are left off by default.

## Evaluation

```bash
python evaluate.py runs/emopatt/best.pt --split test
```

Reports accuracy with test-time augmentation, a confusion matrix, and per-class
precision/recall/F1. Test-time augmentation averages the softmax over two scales with and without a
horizontal flip; `--no_tta` disables it.

Passing several checkpoints soft-votes them, which is how most published FER2013 results above 75%
are constructed:

```bash
python evaluate.py runs/*/best.pt --split test
```

`train_ensemble.sh` trains a small ensemble (different backbones, resolutions and seeds) and scores
it. Any `timm` backbone that exposes `features_only` works via `--backbone`.

## Repository layout

| File | Purpose |
| --- | --- |
| `model.py` | `EmoPAttLite` assembly, preprocessing, parameter groups |
| `elements.py` | STN, SE block, CBAM, separable conv, patch extraction, attention classifier, EMA |
| `data.py` | FER2013 loading, batched GPU augmentation, mixup/CutMix, TTA views |
| `train.py` | Two-stage training loop and CLI |
| `evaluate.py` | Checkpoint evaluation, ensembling, confusion matrix |
| `scheduler.py` | Per-step cosine schedule with linear warmup |
| `dataset/load_dataset.py` | FER2013 CSV to lossless npz |
| `train_ensemble.sh` | Sequential ensemble training driver |

## Citation

```bibtex
@incollection{benseddik2026emopattlite,
  author    = {Ben Seddik, Ismail and Adelekan, Adebowale Emmanuel},
  title     = {EmoPAtt-Lite: Lightweight Facial Emotion Recognition},
  booktitle = {Proceedings of International Conference on Information Technology and Applications},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  pages     = {102--115},
  doi       = {10.1007/978-3-032-21174-3_10},
  isbn      = {978-3-032-21174-3}
}
```
