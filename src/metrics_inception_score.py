#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


class FlatImageFolder(Dataset):
    def __init__(self, folder):
        self.paths = []
        folder = Path(folder)
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.paths.extend(list(folder.glob(ext)))
        self.paths = sorted(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def calculate_inception_score(
    image_folder,
    num_samples=30000,
    batch_size=256,
    splits=10,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FlatImageFolder(image_folder)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {image_folder}")

    n = min(num_samples, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(n)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    model = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        transform_input=False,
    ).to(device)
    model.eval()

    preds = []
    for x in loader:
        x = x.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        preds.append(prob.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    for k in range(splits):
        part = preds[k * len(preds) // splits: (k + 1) * len(preds) // splits]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        kl = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl)))

    return float(np.mean(split_scores)), float(np.std(split_scores))
