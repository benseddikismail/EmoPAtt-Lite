#!/usr/bin/env bash
# Train the EmoPAtt-Lite ensemble members one after another, then score them.
#
# The members differ in backbone, input resolution and seed; that diversity is
# what the soft-vote at the end is buying. Each run is independent, so the
# script can be stopped and restarted from whichever member is missing.
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-./.venv/bin/python}
EPOCHS=${EPOCHS:-300}
mkdir -p logs runs

train() {
  local name=$1; shift
  if [ -f "runs/$name/best.pt" ]; then
    echo "== $name already trained, skipping"
    return
  fi
  echo "== training $name"
  "$PY" -W ignore -u train.py --name "$name" --train_epochs 4 \
    --ft_epochs "$((EPOCHS - 4))" --batch_size 128 "$@" 2>&1 | tee "logs/$name.log"
}

train mnv1_224 --backbone mobilenetv1_100      --img_size 224 --seed 0
train r34_160  --backbone resnet34             --img_size 160 --seed 1
train mnv1_160 --backbone mobilenetv1_100      --img_size 160 --seed 2

echo "== ensemble"
"$PY" -W ignore evaluate.py runs/*/best.pt --split test | tee logs/ensemble.log
