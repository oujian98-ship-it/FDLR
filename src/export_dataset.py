import os
from pathlib import Path

import random
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from utils import get_dataset_from_name

parent_path = Path(__file__).parent.parent
export_folder = parent_path / 'evaluation/celeba_5000'
dataset = get_dataset_from_name("celeba", train=False)

# Initialize an empty dictionary to store samples for each label
samples_per_label = {}

# Assign each sample to its corresponding label
for i in range(len(dataset)):
    sample, label = dataset[i]
    if label not in samples_per_label:
        samples_per_label[label] = []
    samples_per_label[label].append(sample)

# Calculate the number of samples to select from each label
total_samples = 5000
num_labels = len(samples_per_label)
samples_per_label_selected = {label: total_samples // num_labels for label in samples_per_label}

# Randomly fill till total_samples = reached
while sum(samples_per_label_selected.values()) < total_samples:
    key = random.randint(0, len(samples_per_label_selected.keys())-1)
    samples_per_label_selected[key] = samples_per_label_selected[key] + 1

# Randomly select the desired number of samples from each label
selected_samples = []
for label in samples_per_label:
    samples = samples_per_label[label]
    num_samples_selected = samples_per_label_selected[label]
    label_samples = torch.utils.data.random_split(samples, [num_samples_selected, len(samples) - num_samples_selected])[0]
    selected_samples.extend([(sample, label) for sample in label_samples])

# Export the selected samples to a folder
os.makedirs(export_folder, exist_ok=True)

for i, sample in enumerate(selected_samples):
    image, label = sample
    # Save the image with the corresponding label in the name
    image_path = os.path.join(export_folder, f"image_{i}_label{label}.png")
    save_image(image, image_path)

print(samples_per_label_selected)