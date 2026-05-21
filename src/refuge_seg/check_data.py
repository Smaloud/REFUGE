from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from refuge_seg.datasets.refuge_dataset import (
    infer_mask_encoding,
    summarize_mask_mapping,
    validate_mask_mapping,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.load()
        return True, ""
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="REFUGE")
    args = parser.parse_args()

    root = Path(args.data_root)
    bad_files: list[tuple[str, str]] = []

    for split in ["train", "val", "test"]:
        image_dir = root / split / "Images"
        if not image_dir.exists():
            print(f"[skip] {image_dir} does not exist")
            continue
        count = 0
        for path in sorted(image_dir.glob("*.jpg")):
            ok, message = check_image(path)
            count += 1
            if not ok:
                bad_files.append((str(path), message))
        print(f"[done] {split}: checked {count} images")

        mask_dir = root / split / "gts"
        if mask_dir.exists():
            sample_masks = sorted(mask_dir.glob("*.bmp"))[:3]
            for mask_path in sample_masks:
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                values, counts = np.unique(mask, return_counts=True)
                raw_counts = {int(value): int(count) for value, count in zip(values, counts)}
                print(f"  [mask] {mask_path.name}: {raw_counts}")
            encoding = infer_mask_encoding(root, split)
            summary = summarize_mask_mapping(root, split, encoding, max_masks=20)
            print(f"  [encoding] {split}: {encoding}")
            print(f"  [summary] {split}: {summary}")
            validate_mask_mapping(root, split, encoding)

    if bad_files:
        print("\nBad files:")
        for path, message in bad_files:
            print(f"- {path}: {message}")
    else:
        print("\nNo unreadable images found.")


if __name__ == "__main__":
    main()
