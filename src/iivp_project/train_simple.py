from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .data import DigitDataset
from .models import SimpleStrokeCNN

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
VALIDATION_PART = 0.15
SAVE_PATH = Path("models/simple_cnn.pt")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_loaders():
    dataset = DigitDataset(split="train")
    val_count = int(len(dataset) * VALIDATION_PART)
    train_count = len(dataset) - val_count

    generator = torch.Generator().manual_seed(26)
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
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
    device = get_device()
    train_loader, val_loader = make_loaders()
    model = SimpleStrokeCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_accuracy = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}: train loss={train_loss:.4f}, train acc={train_acc:.4f}, val loss={val_loss:.4f}, val acc={val_acc:.4f}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
    print("best validation accuracy:", round(best_val_accuracy, 4))

if __name__ == "__main__":
    main()
