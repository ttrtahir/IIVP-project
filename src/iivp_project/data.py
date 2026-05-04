from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .config import DATA_DIR, IMAGE_SIZE


SOBEL_X = torch.tensor(
    [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]
)
SOBEL_Y = torch.tensor(
    [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]
)


def load_grayscale_image(path):
    image = Image.open(path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return TF.to_tensor(image)


def make_image_views(path):
    raw = load_grayscale_image(path)
    batch = raw.unsqueeze(0)

    edge_x = F.conv2d(batch, SOBEL_X, padding=1)
    edge_y = F.conv2d(batch, SOBEL_Y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6).squeeze(0)
    edge = edge / edge.max().clamp(min=1e-6)

    density = F.avg_pool2d(batch, kernel_size=5, stride=1, padding=2).squeeze(0)

    return torch.cat([raw, edge, density], dim=0)


class DigitDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, split="train"):
        self.data_dir = Path(data_dir)
        self.split = split

        if split == "train":
            self.records = pd.read_csv(self.data_dir / "train.csv")
        else:
            self.records = pd.read_csv(self.data_dir / "test.csv")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image_id = int(row["Id"])

        if self.split == "train":
            label = int(row["Category"])
            image_path = self.data_dir / "train" / "train" / str(label) / f"{image_id}.png"
            image = make_image_views(image_path)
            return image, label

        image_path = self.data_dir / "test" / "test" / f"{image_id}.png"
        image = make_image_views(image_path)
        return image, image_id
