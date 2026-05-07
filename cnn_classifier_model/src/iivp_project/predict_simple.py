from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import DigitDataset
from .models import SimpleStrokeCNN
from .train_simple import BATCH_SIZE, SAVE_PATH, get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT_ROOT.parent / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "simple_submission.csv"


def load_model(device):
    if not SAVE_PATH.exists():
        raise FileNotFoundError(f"model not found: {SAVE_PATH}")

    model = SimpleStrokeCNN().to(device)
    state = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_loader():
    dataset = DigitDataset(split="test")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def predict_rows(model, loader, device):
    rows = []

    for images, image_ids in loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()
        image_ids = image_ids.tolist()

        for image_id, prediction in zip(image_ids, predictions):
            rows.append({"Id": int(image_id), "Category": int(prediction)})

    return rows


def save_submission(rows):
    rows = sorted(rows, key=lambda row: row["Id"])
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Id", "Category"]).to_csv(SUBMISSION_PATH, index=False)


def main():
    device = get_device()
    print("using device:", device)

    model = load_model(device)
    loader = make_loader()
    rows = predict_rows(model, loader, device)

    save_submission(rows)
    print("saved:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
