from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .data import DigitDataset, make_image_views_from_image
from .models import ImprovedHeartFailureCNN

BATCH_SIZE = 128
EPOCHS = 40
LR = 1.2e-3
WEIGHT_DECAY = 5e-4
SEED = 26
VALIDATION_PART = 0.10
WARMUP_EPOCHS = 3
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.05
SAVE_PATH = Path("models/improved_cnn.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class AugmentedDataset(DigitDataset): #and also random erasing
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pil_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=12,
                translate=(0.10, 0.10),
                scale=(0.85, 1.15),
                shear=8,
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.20,
            ),
        ])
        self.tensor_transform = transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0
        )

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image_id = int(row["Id"])

        if self.split == "train":
            label = int(row["Category"])
            path = self.image_path(image_id, label)
            pil = Image.open(path).convert("L")
            if self.augment:
                pil = self.pil_transform(pil)
            views = make_image_views_from_image(pil)
            if self.augment:
                views = self.tensor_transform(views)
            return views, label

        path = self.image_path(image_id)
        pil = Image.open(path).convert("L")
        return make_image_views_from_image(pil), image_id


def split_indices_by_label(records):
    train_indices, val_indices = [], []
    generator = torch.Generator().manual_seed(SEED)
    for label in sorted(records["Category"].unique()):
        idx = records.index[records["Category"] == label].tolist()
        order = torch.randperm(len(idx), generator=generator).tolist()
        idx = [idx[i] for i in order]
        n_val = max(1, int(round(len(idx) * VALIDATION_PART)))
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])
    return train_indices, val_indices


def make_loaders():
    train_data = AugmentedDataset(split="train", augment=True)
    val_data = AugmentedDataset(split="train", augment=False)
    train_idx, val_idx = split_indices_by_label(train_data.records)

    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_data, val_idx),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                ssd[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                ssd[k].copy_(v)


def lr_at(epoch_float):
    # Linear warmup then cosine decay.
    import math
    if epoch_float < WARMUP_EPOCHS:
        return LR * (epoch_float + 1) / WARMUP_EPOCHS
    progress = (epoch_float - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return LR * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def accuracy(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_acc = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        total_loss += loss_fn(out, labels).item()
        total_acc += accuracy(out, labels)
        n += 1
    return total_loss / n, total_acc / n


def main():
    set_seed()
    device = get_device()
    print("device:", device)

    train_loader, val_loader = make_loaders()
    model = ImprovedHeartFailureCNN().to(device)
    ema = EMA(model)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_loader)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = running_acc = 0.0
        for step, (images, labels) in enumerate(train_loader):
            epoch_float = epoch + step / steps_per_epoch
            for g in optimizer.param_groups:
                g["lr"] = lr_at(epoch_float)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(images)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            running_acc += accuracy(out, labels)

        train_loss = running_loss / steps_per_epoch
        train_acc = running_acc / steps_per_epoch
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        ema_val_loss, ema_val_acc = evaluate(ema.shadow, val_loader, loss_fn, device)

        print(
            f"epoch {epoch + 1:02d} | lr={optimizer.param_groups[0]['lr']:.5f} "
            f"| train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"| ema val loss={ema_val_loss:.4f} acc={ema_val_acc:.4f}"
        )

        # Save the better of (raw, ema) on validation.
        candidate_acc = max(val_acc, ema_val_acc)
        if candidate_acc > best_val_acc:
            best_val_acc = candidate_acc
            state = ema.shadow.state_dict() if ema_val_acc >= val_acc else model.state_dict()
            torch.save(state, SAVE_PATH)

    print("best validation accuracy:", round(best_val_acc, 4))
    print("saved:", SAVE_PATH)


if __name__ == "__main__":
    main()
