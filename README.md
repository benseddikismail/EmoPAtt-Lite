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

This writes `dataset/fer2013.npz` holding the raw 48x48 pixels as `uint8`. Pass
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

Passing several checkpoints soft-votes them:

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
