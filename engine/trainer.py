from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    max_grad_norm: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_grad_norm = 0.0
    total_samples = 0
    total_steps = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_grad_norm += float(grad_norm.item())
        total_samples += batch_size
        total_steps += 1
        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
            grad_norm=total_grad_norm / total_steps,
        )
    return total_loss / total_samples, total_correct / total_samples, total_grad_norm / total_steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "eval",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=description, leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)
    return total_loss / total_samples, total_correct / total_samples


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, list[float]]:
    training_config = config["training"]
    paths_config = config["paths"]
    last_checkpoint = checkpoint_path(paths_config)
    best_checkpoint = checkpoint_path(paths_config, best=True)
    start_epoch = 1
    best_val_accuracy = 0.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "grad_norm": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    if training_config.get("resume", True) and last_checkpoint.exists():
        checkpoint = load_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_accuracy = float(checkpoint.get("best_val_accuracy", 0.0))
        history = checkpoint.get("history", history)
        history.setdefault("grad_norm", [])
        tqdm.write(f"Resumed from {last_checkpoint} at epoch {start_epoch}")

    log_dir = Path(paths_config.get("log_dir", "runs/mnist"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    try:
        if start_epoch == 1:
            initial_val_loss, initial_val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
                "val epoch 0",
            )
            best_val_accuracy = initial_val_accuracy
            writer.add_scalars("loss", {"val": initial_val_loss}, 0)
            writer.add_scalars("accuracy", {"val": initial_val_accuracy}, 0)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], 0)
            tqdm.write(
                f"epoch=000 val_loss={initial_val_loss:.4f} "
                f"val_acc={initial_val_accuracy:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                best_val_accuracy=best_val_accuracy,
                history=history,
                config=config,
            )

        for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
            train_loss, train_accuracy, grad_norm = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                use_amp=bool(training_config.get("mixed_precision", True)) and device.type == "cuda",
                max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
            )
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, f"val epoch {epoch}")

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["grad_norm"].append(grad_norm)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars(
                "accuracy",
                {"train": train_accuracy, "val": val_accuracy},
                epoch,
            )
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("grad_norm", grad_norm, epoch)

            tqdm.write(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} grad_norm={grad_norm:.4f}"
            )

            if epoch % int(training_config.get("checkpoint_every", 1)) == 0:
                save_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                    history=history,
                    config=config,
                )
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                    history=history,
                    config=config,
                )
    except KeyboardInterrupt:
        interrupted_epoch = start_epoch + len(history["train_loss"]) - 1
        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=max(interrupted_epoch, 0),
            best_val_accuracy=best_val_accuracy,
            history=history,
            config=config,
        )
        tqdm.write(f"Interrupted. Saved checkpoint to {last_checkpoint}")
    finally:
        writer.close()

    return history
