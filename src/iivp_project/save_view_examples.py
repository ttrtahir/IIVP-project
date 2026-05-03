from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF

from .config import DATA_DIR, IMAGE_SIZE
from .data import make_image_views


OUTPUT_DIR = Path("runs/view_examples")


def train_image_path(data_dir, image_id, label):
    return Path(data_dir) / "train" / "train" / str(label) / f"{image_id}.png"


def tensor_to_image(tensor):
    tensor = tensor.clamp(0, 1)
    return TF.to_pil_image(tensor)


def make_contact_sheet(raw, edge, density):
    sheet = Image.new("L", (IMAGE_SIZE * 3, IMAGE_SIZE))
    sheet.paste(tensor_to_image(raw), (0, 0))
    sheet.paste(tensor_to_image(edge), (IMAGE_SIZE, 0))
    sheet.paste(tensor_to_image(density), (IMAGE_SIZE * 2, 0))
    return sheet


def save_example(image_path, output_path):
    views = make_image_views(image_path)
    raw = views[0:1]
    edge = views[1:2]
    density = views[2:3]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_contact_sheet(raw, edge, density).save(output_path)


def pick_first_examples(records, examples_per_class=2):
    examples = []
    for label in range(10):
        rows = records[records["Category"] == label].head(examples_per_class)
        for _, row in rows.iterrows():
            examples.append((int(row["Id"]), int(row["Category"])))
    return examples


def main():
    data_dir = DATA_DIR
    records = pd.read_csv(Path(data_dir) / "train.csv")
    examples = pick_first_examples(records)

    for image_id, label in examples:
        image_path = train_image_path(data_dir, image_id, label)
        output_path = OUTPUT_DIR / f"class_{label}_id_{image_id}.png"
        save_example(image_path, output_path)

    print("saved raw edge density examples to:", OUTPUT_DIR)
    print("number of examples:", len(examples))


if __name__ == "__main__":
    main()

