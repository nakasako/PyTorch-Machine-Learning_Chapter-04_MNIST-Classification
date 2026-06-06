from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms

from data.mnist import build_eval_transform
from models import build_model, build_model_config
from train import resolve_device
from utils import checkpoint_path, load_checkpoint, load_config

CONFIG_PATH = "configs/train.yaml"
CHECKPOINT_PATH: str | None = None
IMAGE_PATH: str | None = None
TEST_INDEX = 0


def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(image).unsqueeze(0)


def main() -> None:
    config = load_config(CONFIG_PATH)
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model(build_model_config(config["model"])).to(device)
    ckpt_path = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else checkpoint_path(config["paths"], best=True)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    label: int | None = None
    if IMAGE_PATH:
        image_tensor = load_image_tensor(Path(IMAGE_PATH))
    else:
        dataset = datasets.MNIST(
            root=config["data"].get("root", "data/raw"),
            train=False,
            transform=build_eval_transform(),
            download=config["data"].get("download", True),
        )
        image_tensor, label = dataset[TEST_INDEX]
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        prediction = int(probabilities.argmax().item())

    print(f"checkpoint={ckpt_path}")
    if label is not None:
        print(f"label={label}")
    print(f"prediction={prediction}")
    print(f"confidence={probabilities[prediction].item():.4f}")

    plt.imshow(image_tensor.squeeze(0).squeeze(0), cmap="gray")
    title = f"prediction={prediction}"
    if label is not None:
        title += f" label={label}"
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
