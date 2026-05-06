from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from .data import DigitDataset
from .models import SimpleStrokeCNN

BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001
SEED = 26
VALIDATION_PART = 0.15
SAVE_PATH = Path("models/simple_cnn.pt")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def split_indices_by_label(records):
    # Keep validation balanced between digit classes.
    train_indices = []
    val_indices = []

    generator = torch.Generator().manual_seed(SEED)
    for label in sorted(records["Category"].unique()):
        label_indices = records.index[records["Category"] == label].tolist()
        order = torch.randperm(len(label_indices), generator=generator).tolist()
        label_indices = [label_indices[index] for index in order]
        val_count = max(1, int(round(len(label_indices) * VALIDATION_PART)))
        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    return train_indices, val_indices


def make_loaders():
    train_data = DigitDataset(split="train", augment=True)
    val_data = DigitDataset(split="train", augment=False)
    train_indices, val_indices = split_indices_by_label(train_data.records)

    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader

def accuracy(outputs, labels):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)

def run_train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
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
        total_accuracy += accuracy(outputs, labels)
        batch_count += 1
    return total_loss / batch_count, total_accuracy / batch_count

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        total_accuracy += accuracy(outputs, labels)
        batch_count += 1
    return total_loss / batch_count, total_accuracy / batch_count

def main():
    set_seed()
    device = get_device()
    train_loader, val_loader = make_loaders()

    model = SimpleStrokeCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        print(f"epoch {epoch}: train loss={train_loss:.4f}, train acc={train_acc:.4f}, val loss={val_loss:.4f}, val acc={val_acc:.4f}")
        is_better_accuracy = val_acc > best_val_accuracy
        is_lower_loss_tie = val_acc == best_val_accuracy and val_loss < best_val_loss
        if is_better_accuracy or is_lower_loss_tie:
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
    print("best validation accuracy:", round(best_val_accuracy, 4))
    print("best validation loss:", round(best_val_loss, 4))

if __name__ == "__main__":
    main()
