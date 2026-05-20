from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from refuge_seg.datasets.refuge_dataset import REFUGEDataset
from refuge_seg.utils.visualization import colorize_mask, denormalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="REFUGE")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--output", type=str, default="outputs/figures/task2_dataset_overview.png")
    args = parser.parse_args()

    dataset = REFUGEDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    samples = []
    for batch in loader:
        samples.append(batch)
        if len(samples) >= args.num_samples:
            break

    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, batch in enumerate(samples):
        image = denormalize(batch["image"][0])
        mask = batch["mask"][0].numpy()
        disc = (mask == 1).astype(np.uint8) * 255
        cup = (mask == 2).astype(np.uint8) * 255

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image: {batch['id'][0]}")
        axes[row, 1].imshow(colorize_mask(mask))
        axes[row, 1].set_title("Mask")
        axes[row, 2].imshow(disc, cmap="gray")
        axes[row, 2].imshow(cup, cmap="cool", alpha=0.45)
        axes[row, 2].set_title("Disc/Cup")

        for col in range(3):
            axes[row, col].axis("off")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
