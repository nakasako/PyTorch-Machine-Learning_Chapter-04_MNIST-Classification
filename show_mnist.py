from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from utils import load_config

CONFIG_PATH = "configs/train.yaml"
COUNT = 25
SAVE_PATH: str | None = None


def print_dataset_info(dataset: datasets.MNIST) -> None:
    print(f"dataset={dataset.__class__.__name__}")
    print(f"total_samples={len(dataset)}")
    print(f"image_size={dataset.data.shape[-2]}x{dataset.data.shape[-1]}")
    print(f"raw_data_shape={tuple(dataset.data.shape)}")
    print(f"num_classes={len(dataset.targets.unique())}")


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset = datasets.MNIST(
        root=Path(config["data"].get("root", "data/raw")),
        train=True,
        transform=transforms.ToTensor(),
        download=config["data"].get("download", True),
    )
    print_dataset_info(dataset)

    count = min(COUNT, len(dataset))
    cols = 5
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes_list = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for axis in axes_list:
        axis.axis("off")
    for index in range(count):
        image, label = dataset[index]
        axes_list[index].imshow(image.squeeze(0), cmap="gray")
        axes_list[index].set_title(str(label))

    fig.tight_layout()
    if SAVE_PATH:
        output_path = Path(SAVE_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved={output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
