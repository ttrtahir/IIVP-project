from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .config import DATA_DIR, IMAGE_SIZE


TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
AUGMENT_TRANSFORM = {
    "degrees": 10,
    "translate": (0.06, 0.06),
    "scale": (0.92, 1.10),
    "shear": 4,
    "fill": 0,
}
CHANNEL_MEAN = torch.tensor([0.25, 0.16, 0.25]).view(3, 1, 1)
CHANNEL_STD = torch.tensor([0.38, 0.25, 0.34]).view(3, 1, 1)
SOBEL_X = torch.tensor(
    [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]
)
SOBEL_Y = torch.tensor(
    [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]
)


def load_grayscale_image(path):
    """Load one image as a resized grayscale tensor."""
    image = Image.open(path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return TF.to_tensor(image)


def make_edge_view(raw):
    """Create a Sobel edge view from a single-channel image tensor."""
    batch = raw.unsqueeze(0)
    edge_x = F.conv2d(batch, SOBEL_X, padding=1)
    edge_y = F.conv2d(batch, SOBEL_Y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6).squeeze(0)
    return edge / edge.max().clamp(min=1e-6)


def make_density_view(raw):
    """Create a local stroke-density view using average pooling."""
    batch = raw.unsqueeze(0)
    return F.avg_pool2d(batch, kernel_size=5, stride=1, padding=2).squeeze(0)


def normalize_views(views):
    return (views - CHANNEL_MEAN) / CHANNEL_STD


def make_image_views_from_image(image):
    image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    raw = TF.to_tensor(image)

    # Three views: raw digit, Sobel edges, and local stroke density.
    edge = make_edge_view(raw)
    density = make_density_view(raw)
    views = torch.cat([raw, edge, density], dim=0)
    return normalize_views(views)


def make_image_views(path):
    image = Image.open(path).convert("L")
    return make_image_views_from_image(image)


class DigitDataset(Dataset):
    """Dataset for train/test digit images."""

    def __init__(self, data_dir=DATA_DIR, split=TRAIN_SPLIT, augment=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.transform = transforms.RandomAffine(**AUGMENT_TRANSFORM)

        if split == TRAIN_SPLIT:
            self.records = pd.read_csv(self.data_dir / "train.csv")
        else:
            self.records = pd.read_csv(self.data_dir / "test.csv")

    def image_path(self, image_id, label=None):
        if self.split == TRAIN_SPLIT:
            return self.data_dir / "train" / "train" / str(label) / f"{image_id}.png"
        return self.data_dir / "test" / "test" / f"{image_id}.png"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image_id = int(row["Id"])

        if self.split == TRAIN_SPLIT:
            label = int(row["Category"])
            image_path = self.image_path(image_id, label)
            if self.augment:
                pil_image = Image.open(image_path).convert("L")
                pil_image = self.transform(pil_image)
                image = make_image_views_from_image(pil_image)
                return image, label
            image = make_image_views(image_path)
            return image, label

        image_path = self.image_path(image_id)
        image = make_image_views(image_path)
        return image, image_id
