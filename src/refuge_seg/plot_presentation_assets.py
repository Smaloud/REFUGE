from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

from refuge_seg.datasets.refuge_dataset import (
    REFUGEDataset,
    infer_mask_encoding,
    summarize_mask_mapping,
)
from refuge_seg.utils.postprocess import diagnose_prediction, postprocess_prediction
from refuge_seg.utils.visualization import colorize_mask, denormalize

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _save(fig: plt.Figure, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_label_sanity(data_root: str | Path, output: str | Path, num_samples: int = 4) -> None:
    data_root = Path(data_root)
    dataset = REFUGEDataset(data_root, "train", image_size=512, augment=False)
    encoding = infer_mask_encoding(data_root, "train", max_masks=50)
    summary = summarize_mask_mapping(data_root, "train", encoding, max_masks=50)

    rows = min(num_samples, len(dataset))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(rows):
        sample = dataset[row]
        image_path = data_root / "train" / "Images" / f"{sample['id']}.jpg"
        mask_path = data_root / "train" / "gts" / f"{sample['id']}.bmp"
        raw_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        values, counts = np.unique(raw_mask, return_counts=True)
        raw_counts = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))

        axes[row, 0].imshow(np.array(Image.open(image_path).convert("RGB")))
        axes[row, 0].set_title(f"Image {sample['id']}")
        axes[row, 1].imshow(raw_mask, cmap="gray")
        axes[row, 1].set_title(f"Raw mask\n{raw_counts}")
        axes[row, 2].imshow(colorize_mask(sample["mask"].numpy()))
        axes[row, 2].set_title("Mapped mask")

        labels = ["background", "disc rim", "cup"]
        ratios = [
            summary["background_ratio"],
            summary["disc_rim_ratio"],
            summary["cup_ratio"],
        ]
        axes[row, 3].bar(labels, ratios, color=["#222222", "#ffb400", "#00d5d5"])
        axes[row, 3].set_ylim(0, 1)
        axes[row, 3].set_title(
            "Inferred encoding\n"
            f"bg={encoding.background}, rim={encoding.disc_rim}, cup={encoding.cup}"
        )
        axes[row, 3].tick_params(axis="x", rotation=20)

        for col in range(3):
            axes[row, col].axis("off")

    _save(fig, output)


def _ellipse_mask(height: int, width: int, center: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy, cx = center
    ry, rx = radii
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def plot_postprocess_flow(output: str | Path) -> None:
    pred = np.zeros((160, 220), dtype=np.uint8)
    disc = _ellipse_mask(160, 220, (78, 82), (42, 34))
    cup = _ellipse_mask(160, 220, (78, 84), (22, 18))
    pred[disc] = 1
    pred[cup] = 2

    pred[_ellipse_mask(160, 220, (80, 84), (8, 7))] = 0
    pred[_ellipse_mask(160, 220, (40, 170), (10, 9))] = 1
    pred[_ellipse_mask(160, 220, (122, 170), (9, 8))] = 2

    processed = postprocess_prediction(pred)
    before_diag = diagnose_prediction(pred)
    after_diag = diagnose_prediction(processed)
    changed = np.zeros((*pred.shape, 3), dtype=np.uint8)
    changed[(pred != processed) & (processed == 0)] = [200, 40, 40]
    changed[(pred != processed) & (processed > 0)] = [40, 160, 80]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(colorize_mask(pred))
    axes[0].set_title(
        "Raw prediction\n"
        f"disc cc={before_diag['disc_components']}, cup cc={before_diag['cup_components']}"
    )
    axes[1].imshow((pred == 1) | (pred == 2), cmap="gray")
    axes[1].set_title("Disc union mask\nwith fragments/hole")
    axes[2].imshow(colorize_mask(processed))
    axes[2].set_title(
        "After postprocess\n"
        f"disc cc={after_diag['disc_components']}, cup cc={after_diag['cup_components']}"
    )
    axes[3].imshow(changed)
    axes[3].set_title("Changed pixels\nred=removed, green=filled")
    for ax in axes:
        ax.axis("off")
    _save(fig, output)


def _load_prediction_summary(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data = json.load(path.open("r", encoding="utf-8"))
    if "class_distribution_examples" in data:
        return data["class_distribution_examples"]
    return data


def plot_prediction_diagnostics(summary_paths: list[Path], output: str | Path) -> None:
    fig, axes = plt.subplots(1, max(len(summary_paths), 1), figsize=(6 * max(len(summary_paths), 1), 4))
    axes = np.atleast_1d(axes)

    if not summary_paths:
        axes[0].text(0.5, 0.5, "No prediction_summary.json provided", ha="center", va="center")
        axes[0].axis("off")
        _save(fig, output)
        return

    for ax, path in zip(axes, summary_paths):
        summary = _load_prediction_summary(path)
        if not summary:
            ax.text(0.5, 0.5, f"Missing\n{path}", ha="center", va="center", fontsize=9)
            ax.axis("off")
            continue
        foreground_ratios = []
        cup_ratios = []
        for counts in summary.values():
            numeric = {int(k): int(v) for k, v in counts.items()}
            total = sum(numeric.values())
            if total == 0:
                continue
            foreground_ratios.append((numeric.get(1, 0) + numeric.get(2, 0)) / total)
            cup_ratios.append(numeric.get(2, 0) / total)
        ax.hist(foreground_ratios, bins=20, alpha=0.7, label="disc+cup")
        ax.hist(cup_ratios, bins=20, alpha=0.7, label="cup")
        ax.set_title(f"{path.parent.parent.name}\n{len(summary)} samples")
        ax.set_xlabel("Predicted area ratio")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="REFUGE")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--summary", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_label_sanity(args.data_root, output_dir / "task2_label_sanity.png")
    plot_postprocess_flow(output_dir / "task2_postprocess_flow.png")
    summary_paths = [Path(path) for path in args.summary]
    plot_prediction_diagnostics(summary_paths, output_dir / "task2_prediction_diagnostics.png")


if __name__ == "__main__":
    main()
