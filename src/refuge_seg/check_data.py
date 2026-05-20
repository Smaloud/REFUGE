from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFile

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

    if bad_files:
        print("\nBad files:")
        for path, message in bad_files:
            print(f"- {path}: {message}")
    else:
        print("\nNo unreadable images found.")


if __name__ == "__main__":
    main()
