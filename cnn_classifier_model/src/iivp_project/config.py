from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_data_dir():
    # Check both this folder and the main project folder.
    choices = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
        PROJECT_ROOT / "iivp-2026-challenge",
        PROJECT_ROOT.parent / "iivp-2026-challenge",
        Path.home() / "Downloads" / "iivp-2026-challenge",
    ]

    for path in choices:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path

    return choices[0]


DATA_DIR = find_data_dir()
IMAGE_SIZE = 32
NUM_CLASSES = 10
