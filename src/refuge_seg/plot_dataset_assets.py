from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_dataset_stats(root: Path, output: Path) -> None:
    stats = []
    for split in ["train", "val", "test"]:
        image_dir = root / split / "Images"
        mask_dir = root / split / "gts"
        images = sorted(image_dir.glob("*.jpg")) if image_dir.exists() else []
        masks = sorted(mask_dir.glob("*.bmp")) if mask_dir.exists() else []
        stats.append((split, len(images), len(masks)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    splits = [item[0] for item in stats]
    image_counts = [item[1] for item in stats]
    mask_counts = [item[2] for item in stats]
    x = np.arange(len(splits))
    width = 0.35
    axes[0].bar(x - width / 2, image_counts, width=width, label="images")
    axes[0].bar(x + width / 2, mask_counts, width=width, label="masks")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_title("Dataset Split Counts")
    axes[0].set_ylabel("count")
    axes[0].legend()

    sample_paths = sorted((root / "train" / "gts").glob("*.bmp"))[:50]
    pixel_counts = {255: 0, 128: 0, 0: 0}
    for path in sample_paths:
        image = Image.open(path)
        image.load()
        mask = np.array(image.convert("L"))
        uniq, cnt = np.unique(mask, return_counts=True)
        for value, count in zip(uniq, cnt):
            pixel_counts[int(value)] = pixel_counts.get(int(value), 0) + int(count)

    labels = ["background(255)", "disc(128)", "cup(0)"]
    values = [pixel_counts.get(255, 0), pixel_counts.get(128, 0), pixel_counts.get(0, 0)]
    axes[1].pie(values, labels=labels, autopct="%1.1f%%")
    axes[1].set_title("Pixel Ratio in First 50 Train Masks")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_structure_example(root: Path, output: Path) -> None:
    image_path = sorted((root / "train" / "Images").glob("*.jpg"))[0]
    mask_path = root / "train" / "gts" / f"{image_path.stem}.bmp"
    image_file = Image.open(image_path)
    image_file.load()
    image = np.array(image_file.convert("RGB"))
    mask_file = Image.open(mask_path)
    mask_file.load()
    mask = np.array(mask_file.convert("L"))
    disc = (mask == 128).astype(np.uint8)
    cup = (mask == 0).astype(np.uint8)

    overlay = image.copy().astype(np.float32)
    overlay[disc == 1] = overlay[disc == 1] * 0.5 + np.array([255, 200, 0], dtype=np.float32) * 0.5
    overlay[cup == 1] = overlay[cup == 1] * 0.4 + np.array([0, 255, 255], dtype=np.float32) * 0.6

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("Fundus Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Raw Mask")
    axes[2].imshow(disc, cmap="gray")
    axes[2].set_title("Optic Disc")
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="REFUGE")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stats", "structure", "all"],
        default="all",
    )
    parser.add_argument("--stats-output", type=str, default="outputs/figures/task2_dataset_stats.png")
    parser.add_argument("--structure-output", type=str, default="outputs/figures/task2_structure_example.png")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.mode in {"stats", "all"}:
        build_dataset_stats(root, Path(args.stats_output))
    if args.mode in {"structure", "all"}:
        build_structure_example(root, Path(args.structure_output))


if __name__ == "__main__":
    main()
