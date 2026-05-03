from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .config import DATA_DIR, NUM_CLASSES
from .data import DigitDataset
from .utils import count_labels, print_label_counts


def train_image_path(data_dir, image_id, label):
    return Path(data_dir) / "train" / "train" / str(label) / f"{image_id}.png"


def test_image_path(data_dir, image_id):
    return Path(data_dir) / "test" / "test" / f"{image_id}.png"


def load_csvs(data_dir):
    train = pd.read_csv(Path(data_dir) / "train.csv")
    test = pd.read_csv(Path(data_dir) / "test.csv")
    return train, test


def print_csv_info(train, test):
    print("train rows:", len(train))
    print("test rows:", len(test))
    print("train columns:", list(train.columns))
    print("test columns:", list(test.columns))


def print_class_balance(train):
    counts = count_labels(train["Category"].tolist())
    print("label counts:")
    print_label_counts(counts)


def check_labels(train):
    labels = sorted(train["Category"].unique().tolist())
    expected = list(range(NUM_CLASSES))
    if labels == expected:
        print("labels look correct:", labels)
    else:
        print("labels are different than expected:", labels)


def check_train_files(data_dir, train, limit=20):
    missing = []
    for _, row in train.head(limit).iterrows():
        image_id = int(row["Id"])
        label = int(row["Category"])
        path = train_image_path(data_dir, image_id, label)
        if not path.exists():
            missing.append(str(path))
    print_missing_or_ok("train", missing, limit)


def check_test_files(data_dir, test, limit=20):
    missing = []
    for _, row in test.head(limit).iterrows():
        image_id = int(row["Id"])
        path = test_image_path(data_dir, image_id)
        if not path.exists():
            missing.append(str(path))
    print_missing_or_ok("test", missing, limit)


def print_missing_or_ok(split, missing, limit):
    if missing:
        print("missing", split, "examples:")
        for path in missing:
            print(path)
    else:
        print("checked", split, "image paths:", limit)


def check_loader_shapes():
    train_data = DigitDataset(split="train")
    test_data = DigitDataset(split="test")

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    train_images, labels = next(iter(train_loader))
    test_images, image_ids = next(iter(test_loader))

    print("train image batch:", train_images.shape)
    print("train label batch:", labels.shape)
    print("test image batch:", test_images.shape)
    print("test id batch:", image_ids.shape)


def main():
    data_dir = DATA_DIR
    train, test = load_csvs(data_dir)
    print_csv_info(train, test)
    print_class_balance(train)
    check_labels(train)
    check_train_files(data_dir, train)
    check_test_files(data_dir, test)
    check_loader_shapes()


if __name__ == "__main__":
    main()

