from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .config import DATA_DIR, IMAGE_SIZE
from .data import make_image_views_from_image
from .models import ImprovedHeartFailureCNN
from .train_improved import BATCH_SIZE, SAVE_PATH, get_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT_ROOT.parent / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "improved_submission.csv"


class TestPILDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.records = pd.read_csv(Path(data_dir) / "test.csv")
        self.test_dir = Path(data_dir) / "test" / "test"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id = int(self.records.iloc[index]["Id"])
        pil = Image.open(self.test_dir / f"{image_id}.png").convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        return pil_to_views(pil), image_id


def pil_to_views(pil):
    return make_image_views_from_image(pil)


def tta_variants(pil):
    """Return a list of view tensors for the same image under small TTA."""
    out = [make_image_views_from_image(pil)]
    for angle in (-7, 7):
        rotated = TF.rotate(pil, angle, fill=0)
        out.append(make_image_views_from_image(rotated))
    return out


class TTADataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.records = pd.read_csv(Path(data_dir) / "test.csv")
        self.test_dir = Path(data_dir) / "test" / "test"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id = int(self.records.iloc[index]["Id"])
        pil = Image.open(self.test_dir / f"{image_id}.png").convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        variants = tta_variants(pil)
        # Stack on a new tta-axis -> (T, C, H, W)
        return torch.stack(variants, dim=0), image_id


def load_model(device, path=SAVE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    model = ImprovedHeartFailureCNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_rows(model, loader, device):
    rows = []
    for batch, image_ids in loader:
        # batch: (B, T, C, H, W)
        b, t, c, h, w = batch.shape
        batch = batch.view(b * t, c, h, w).to(device, non_blocking=True)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).view(b, t, -1).mean(dim=1)
        preds = probs.argmax(dim=1).cpu().tolist()
        for image_id, pred in zip(image_ids.tolist(), preds):
            rows.append({"Id": int(image_id), "Category": int(pred)})
    return rows


def save_submission(rows, path=SUBMISSION_PATH):
    rows = sorted(rows, key=lambda r: r["Id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Id", "Category"]).to_csv(path, index=False)


def main():
    device = get_device()
    print("device:", device)
    model = load_model(device)
    loader = DataLoader(TTADataset(), batch_size=BATCH_SIZE, shuffle=False)
    rows = predict_rows(model, loader, device)
    save_submission(rows)
    print("saved:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
