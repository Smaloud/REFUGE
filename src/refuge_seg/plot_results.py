from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(histories: dict[str, dict[str, list[float]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, history in histories.items():
        axes[0].plot(history["train_loss"], label=f"{label}-train")
        axes[0].plot(history["val_loss"], label=f"{label}-val")
        axes[1].plot(history["val_mean_dice"], label=label)

    axes[0].set_title("训练/验证损失曲线")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("验证集 Mean Dice")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Mean Dice")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_metrics_bar(metrics: dict[str, dict[str, float]], output: Path) -> None:
    labels = list(metrics.keys())
    keys = ["dice_disc", "dice_cup", "iou_disc", "iou_cup", "mean_dice", "mean_iou"]
    values = np.array([[metrics[label].get(key, 0.0) for key in keys] for label in labels])

    x = np.arange(len(keys))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for idx, label in enumerate(labels):
        ax.bar(x + idx * width - width * (len(labels) - 1) / 2, values[idx], width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("不同实验配置的指标对比")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", nargs=3, metavar=("NAME", "HISTORY", "METRICS"))
    parser.add_argument("--curve-output", type=str, default="outputs/figures/task2_training_curves.png")
    parser.add_argument("--bar-output", type=str, default="outputs/figures/task2_metrics_bar.png")
    args = parser.parse_args()

    histories = {}
    metrics = {}
    for name, history_path, metrics_path in args.experiment or []:
        histories[name] = load_history(Path(history_path))
        metrics[name] = load_metrics(Path(metrics_path))

    if histories:
        plot_training_curves(histories, Path(args.curve_output))
    if metrics:
        plot_metrics_bar(metrics, Path(args.bar_output))


if __name__ == "__main__":
    main()

