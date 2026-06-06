from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    mlp: dict
    cnn: dict


def build_model_config(config: dict) -> ModelConfig:
    return ModelConfig(
        name=str(config.get("name", "mlp")).lower(),
        num_classes=int(config.get("num_classes", 10)),
        mlp=dict(config.get("mlp", {})),
        cnn=dict(config.get("cnn", {})),
    )


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        input_dim = 28 * 28
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            group_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            group_norm(c1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            group_norm(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            group_norm(c2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_model(config: ModelConfig | dict) -> nn.Module:
    model_config = build_model_config(config) if isinstance(config, dict) else config
    if model_config.name == "mlp":
        model = MLPClassifier(
            num_classes=model_config.num_classes,
            hidden_dims=list(model_config.mlp.get("hidden_dims", [256, 128])),
            dropout=float(model_config.mlp.get("dropout", 0.2)),
        )
    elif model_config.name == "cnn":
        model = CNNClassifier(
            num_classes=model_config.num_classes,
            channels=list(model_config.cnn.get("channels", [32, 64])),
            dropout=float(model_config.cnn.get("dropout", 0.25)),
        )
    else:
        raise ValueError(f"Unknown model name: {model_config.name}")
    initialize_weights(model)
    return model
