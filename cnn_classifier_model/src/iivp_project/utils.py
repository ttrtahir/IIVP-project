"""Small shared helpers used by the CNN training and prediction scripts."""

import torch


def get_device(prefer_mps=False):
    """Pick the best available PyTorch device."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Set the PyTorch random seed used by the training scripts."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(outputs, labels):
    """Return batch accuracy for classification logits."""
    predictions = outputs.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def split_indices_by_label(records, validation_part, seed):
    """Create a balanced train/validation split from a label dataframe."""
    train_indices = []
    val_indices = []
    generator = torch.Generator().manual_seed(seed)

    for label in sorted(records["Category"].unique()):
        label_indices = records.index[records["Category"] == label].tolist()
        order = torch.randperm(len(label_indices), generator=generator).tolist()
        label_indices = [label_indices[index] for index in order]

        val_count = max(1, int(round(len(label_indices) * validation_part)))
        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    return train_indices, val_indices
