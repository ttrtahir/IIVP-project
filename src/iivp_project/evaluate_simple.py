from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import DigitDataset
from .models import SimpleStrokeCNN
from .train_simple import BATCH_SIZE, SAVE_PATH, VALIDATION_PART
from .utils import NUM_CLASSES, get_device, write_csv


CONFUSION_PATH = Path("runs/simple_confusion.csv")
MISTAKES_PATH = Path("runs/simple_mistakes.csv")


def make_validation_loader():
    dataset = DigitDataset(split="train")
    val_count = int(len(dataset) * VALIDATION_PART)
    train_count = len(dataset) - val_count

    generator = torch.Generator().manual_seed(26)
    _, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader


def load_saved_model(device):
    if not SAVE_PATH.exists():
        raise FileNotFoundError(f"model not found: {SAVE_PATH}")

    model = SimpleStrokeCNN().to(device)
    state = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def new_confusion_matrix():
    return [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]


def add_predictions_to_matrix(matrix, outputs, labels):
    predictions = outputs.argmax(dim=1).detach().cpu().tolist()
    labels = labels.detach().cpu().tolist()

    for actual, predicted in zip(labels, predictions):
        matrix[int(actual)][int(predicted)] += 1


def count_correct(outputs, labels):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def make_mistake_rows(outputs, labels, start_row):
    predictions = outputs.argmax(dim=1).detach().cpu().tolist()
    labels = labels.detach().cpu().tolist()
    rows = []

    for offset, (actual, predicted) in enumerate(zip(labels, predictions)):
        if int(actual) != int(predicted):
            rows.append(
                {
                    "row": start_row + offset,
                    "actual": int(actual),
                    "predicted": int(predicted),
                }
            )

    return rows


def matrix_to_rows(matrix):
    rows = []
    for actual, values in enumerate(matrix):
        row = {"actual": actual}
        for predicted, count in enumerate(values):
            row[f"pred_{predicted}"] = count
        rows.append(row)
    return rows


def save_confusion_matrix(matrix):
    fieldnames = ["actual"]
    for label in range(NUM_CLASSES):
        fieldnames.append(f"pred_{label}")

    rows = matrix_to_rows(matrix)
    write_csv(CONFUSION_PATH, fieldnames, rows)


def save_mistakes(rows):
    fieldnames = ["row", "actual", "predicted"]
    write_csv(MISTAKES_PATH, fieldnames, rows)


@torch.no_grad()
def evaluate(model, loader, device):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    current_row = 0
    mistakes = []
    matrix = new_confusion_matrix()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += count_correct(outputs, labels)
        total_images += batch_size

        add_predictions_to_matrix(matrix, outputs, labels)
        mistakes.extend(make_mistake_rows(outputs, labels, current_row))
        current_row += batch_size

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    return average_loss, accuracy, matrix, mistakes


def main():
    device = get_device()
    print("using device:", device)

    model = load_saved_model(device)
    loader = make_validation_loader()
    loss, accuracy, matrix, mistakes = evaluate(model, loader, device)

    save_confusion_matrix(matrix)
    save_mistakes(mistakes)

    print("validation loss:", round(loss, 4))
    print("validation accuracy:", round(accuracy, 4))
    print("validation images:", len(loader.dataset))
    print("mistakes:", len(mistakes))
    print("saved confusion matrix:", CONFUSION_PATH)
    print("saved mistakes:", MISTAKES_PATH)
    print("finished evaluation")


if __name__ == "__main__":
    main()
