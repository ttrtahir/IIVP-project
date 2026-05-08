import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import DigitDataset
from .model import StrokeViewCNN
from .utils import accuracy, get_device, set_seed, split_indices_by_label


BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 0.0001
SEED = 26
VALIDATION_PART = 0.15

OUTPUT_DIR = Path("runs/cnn")
SAVE_PATH = Path("models/final_cnn.pt")
METRICS_PATH = OUTPUT_DIR / "metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def make_loaders():
    train_data = DigitDataset(split="train", augment=True)
    val_data = DigitDataset(split="train", augment=False)
    train_indices, val_indices = split_indices_by_label(
        train_data.records, VALIDATION_PART, SEED
    )

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_data, val_indices),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def run_train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)
        batch_count += 1

    return total_loss / batch_count, total_acc / batch_count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)
        batch_count += 1

    return total_loss / batch_count, total_acc / batch_count


def save_checkpoint(model, best_val_accuracy, best_val_loss, device):
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "StrokeViewCNN",
            "num_classes": 10,
            "val_accuracy": best_val_accuracy,
            "val_loss": best_val_loss,
            "args": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "val_size": VALIDATION_PART,
                "seed": SEED,
                "device": str(device),
            },
        },
        SAVE_PATH,
    )


def main():
    set_seed(SEED)
    device = get_device(prefer_mps=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders()
    model = StrokeViewCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"epoch {epoch}: train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        better_accuracy = val_acc > best_val_accuracy
        same_accuracy_lower_loss = val_acc == best_val_accuracy and val_loss < best_val_loss
        if better_accuracy or same_accuracy_lower_loss:
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            save_checkpoint(model, best_val_accuracy, best_val_loss, device)

    with METRICS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "epochs": EPOCHS,
                "device": str(device),
            },
            indent=2,
        )
    )
    print("best validation accuracy:", round(best_val_accuracy, 4))
    print("best validation loss:", round(best_val_loss, 4))
    print("saved:", SAVE_PATH)


if __name__ == "__main__":
    main()
