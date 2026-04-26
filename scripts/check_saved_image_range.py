#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="exports/cifar10/dataset_train")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    folder = Path(args.folder)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(ext))
    paths = sorted(paths)[:args.limit]

    assert paths, f"No images found in {folder}"

    for p in paths:
        img = np.array(Image.open(p))
        print(p.name, img.min(), img.max(), float(img.mean()))
        assert img.min() >= 0 and img.max() <= 255

    print("Saved image range OK.")


if __name__ == "__main__":
    main()
