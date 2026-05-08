from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import DATA_DIR, SUBMISSIONS_DIR
from .data import DigitDataset
from .model import StrokeViewCNN
from .utils import get_device


BATCH_SIZE = 128
MODEL_PATH = Path("models/final_cnn.pt")
SUBMISSION_PATH = SUBMISSIONS_DIR / "final_submission.csv"


def load_model(device, path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"model not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)

    model = StrokeViewCNN().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_rows(model, loader, device):
    rows = []
    for images, image_ids in loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1).cpu().tolist()
        for image_id, prediction in zip(image_ids.tolist(), predictions):
            rows.append({"Id": int(image_id), "Category": int(prediction)})
    return rows


def save_submission(rows, path=SUBMISSION_PATH):
    rows = sorted(rows, key=lambda row: row["Id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Id", "Category"]).to_csv(path, index=False)


def main():
    device = get_device(prefer_mps=True)
    print("device:", device)

    model = load_model(device)
    dataset = DigitDataset(data_dir=DATA_DIR, split="test", augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    rows = predict_rows(model, loader, device)
    save_submission(rows)
    print("saved:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
