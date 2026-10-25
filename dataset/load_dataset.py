"""Prepare FER2013 for training.

The CSV ships the raw 48x48 grayscale pixels. We keep them lossless in a single
npz archive instead of re-encoding them as JPEGs: decoding 35k tiny JPEGs every
epoch is both slow and destructive (JPEG artifacts on a 48x48 face are not
negligible). Pass --images if the ImageFolder layout is wanted as well.
"""

import argparse
import os
import tarfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def extract_csv():
    csv_path = os.path.join(HERE, "fer2013", "fer2013.csv")
    if not os.path.exists(csv_path):
        print("Extracting fer2013.tar.gz ...")
        with tarfile.open(os.path.join(HERE, "fer2013.tar.gz"), "r") as tar:
            tar.extractall(HERE)
    return csv_path


def write_images(images, labels, split):
    from PIL import Image

    root = os.path.join(HERE, "FER2013", split)
    for label in EMOTION_LABELS:
        os.makedirs(os.path.join(root, label), exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        path = os.path.join(root, EMOTION_LABELS[label], f"{i}.png")
        Image.fromarray(img, "L").save(path)


def main(images=False):
    csv_path = extract_csv()
    print("Reading", csv_path)
    df = pd.read_csv(csv_path)

    out = {}
    for usage, split in USAGE_TO_SPLIT.items():
        rows = df[df["Usage"] == usage]
        pixels = np.stack(
            [np.fromstring(p, dtype=np.uint8, sep=" ") for p in rows["pixels"]]
        ).reshape(-1, 48, 48)
        labels = rows["emotion"].to_numpy().astype(np.int64)
        out[f"{split}_x"] = pixels
        out[f"{split}_y"] = labels
        counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
        print(
            f"{split:5s} {len(labels):6d} images  "
            + "  ".join(f"{n}:{c}" for n, c in zip(EMOTION_LABELS, counts))
        )
        if images:
            write_images(pixels, labels, split)

    npz_path = os.path.join(HERE, "fer2013.npz")
    np.savez_compressed(npz_path, classes=np.array(EMOTION_LABELS), **out)
    print("Wrote", npz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        action="store_true",
        help="also export the ImageFolder layout as lossless PNGs",
    )
    main(**vars(parser.parse_args()))
