from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


COLORS = np.array(
    [
        [0, 0, 0],
        [255, 180, 0],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return COLORS[mask]


def save_prediction_grid(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(denormalize(image))
    axes[0].set_title("Image")
    axes[1].imshow(colorize_mask(target.cpu().numpy()))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(colorize_mask(pred.cpu().numpy()))
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["val_mean_dice"], label="mean_dice")
    axes[1].set_title("Validation Mean Dice")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

