from torch.utils.data import DataLoader

from .data import DigitDataset
from .models import SimpleStrokeCNN


def main():
    dataset = DigitDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    images, labels = next(iter(loader))
    model = SimpleStrokeCNN()
    outputs = model(images)

    print("image batch shape:", images.shape)
    print("label batch shape:", labels.shape)
    print("model output shape:", outputs.shape)


if __name__ == "__main__":
    main()
