from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    root: Path
    batch_size: int
    num_workers: int
    validation_split: float
    pin_memory: bool
    download: bool


def build_data_config(config: dict) -> DataConfig:
    return DataConfig(
        root=Path(config.get("root", "data/raw")),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 2)),
        validation_split=float(config.get("validation_split", 0.1)),
        pin_memory=bool(config.get("pin_memory", True)),
        download=bool(config.get("download", True)),
    )


def get_mnist_normalize() -> transforms.Normalize:
    return transforms.Normalize((0.1307,), (0.3081,))


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            get_mnist_normalize(),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), get_mnist_normalize()])


def build_dataloaders(config: DataConfig, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_full = datasets.MNIST(
        root=config.root,
        train=True,
        transform=build_train_transform(),
        download=config.download,
    )
    eval_full = datasets.MNIST(
        root=config.root,
        train=True,
        transform=build_eval_transform(),
        download=config.download,
    )
    test_dataset = datasets.MNIST(
        root=config.root,
        train=False,
        transform=build_eval_transform(),
        download=config.download,
    )

    val_size = int(len(train_full) * config.validation_split)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(eval_full, val_indices)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    train_loader = DataLoader(train_subset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
