from pathlib import Path
import csv

import torch
from .config import NUM_CLASSES


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_parent_folder(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    make_parent_folder(path)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def accuracy_from_outputs(outputs, labels):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def count_labels(labels):
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for label in labels:
        counts[int(label)] += 1
    return counts


def print_label_counts(counts):
    for label in range(NUM_CLASSES):
        print(f"class {label}: {counts[label]}")
