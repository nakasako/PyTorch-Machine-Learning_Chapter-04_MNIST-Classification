from __future__ import annotations

import math

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import build_data_config, build_dataloaders
from engine import evaluate, train
from models import build_model, build_model_config
from utils import checkpoint_path, load_checkpoint, load_config, seed_everything

CONFIG_PATH = "configs/train.yaml"
EVAL_ONLY = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_optimizer(model: nn.Module, config: dict) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: AdamW,
    config: dict,
    steps_per_epoch: int,
) -> LambdaLR:
    total_steps = max(1, int(config["epochs"]) * steps_per_epoch)
    warmup_steps = max(0, int(config.get("warmup_epochs", 1)) * steps_per_epoch)
    min_lr_ratio = float(config.get("min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_ratio, (step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    config = load_config(CONFIG_PATH)
    seed_everything(int(config.get("seed", 1234)))
    device = resolve_device(str(config.get("device", "auto")))

    data_config = build_data_config(config["data"])
    train_loader, val_loader, test_loader = build_dataloaders(data_config, int(config.get("seed", 1234)))

    model = build_model(build_model_config(config["model"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"], len(train_loader))
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    if EVAL_ONLY:
        best_path = checkpoint_path(config["paths"], best=True)
        load_checkpoint(best_path, model=model, device=device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, "test")
        print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
        return

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=config,
    )

    best_path = checkpoint_path(config["paths"], best=True)
    load_checkpoint(best_path, model=model, device=device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, "test")
    print(f"best_checkpoint={best_path}")
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
