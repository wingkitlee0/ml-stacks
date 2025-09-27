import ray
import torch


def create_dataset(name: str) -> ray.data.Dataset:
    """Create a dataset for the given name."""
    num_rows = 100 if name == "train" else 10

    return ray.data.from_items(
        [{"x": torch.randn(10), "y": torch.randn(10)} for _ in range(num_rows)]
    )


def create_train_val_datasets() -> tuple[ray.data.Dataset, ray.data.Dataset]:
    train_ds = create_dataset("train")
    val_ds = create_dataset("val")

    return train_ds, val_ds
