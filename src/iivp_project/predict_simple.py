from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DigitDataset
from .models import SimpleStrokeCNN
from .train_simple import BATCH_SIZE, SAVE_PATH
from .utils import count_labels, get_device, print_label_counts, write_csv


SUBMISSION_PATH = Path("submissions/simple_submission.csv")


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


def to_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


@torch.no_grad()
def predict_rows(model, loader, device):
    rows = []
    predictions_for_count = []

    for images, image_ids in loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()
        image_ids = to_list(image_ids)

        for image_id, prediction in zip(image_ids, predictions):
            rows.append({"Id": int(image_id), "Category": int(prediction)})
            predictions_for_count.append(int(prediction))

    return rows, predictions_for_count


def save_submission(rows):
    rows = sorted(rows, key=lambda row: row["Id"])
    write_csv(SUBMISSION_PATH, ["Id", "Category"], rows)


def print_prediction_summary(rows, predictions):
    print("saved submission:", SUBMISSION_PATH)
    print("number of rows:", len(rows))
    print("prediction counts:")
    print_label_counts(count_labels(predictions))


def main():
    device = get_device()
    print("using device:", device)

    model = load_model(device)
    loader = make_loader()
    rows, predictions = predict_rows(model, loader, device)

    save_submission(rows)
    print_prediction_summary(rows, predictions)


if __name__ == "__main__":
    main()

