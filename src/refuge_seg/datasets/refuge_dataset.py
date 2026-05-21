from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class MaskEncoding:
    background: int
    disc_rim: int = 128
    cup: int = 0

    @property
    def raw_to_class(self) -> dict[int, int]:
        return {
            self.background: 0,
            self.disc_rim: 1,
            self.cup: 2,
        }

    @property
    def class_to_raw(self) -> dict[int, int]:
        return {
            0: self.background,
            1: self.disc_rim,
            2: self.cup,
        }


@dataclass
class DatasetConfig:
    data_root: str
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 4


def _mask_counts(mask_paths: list[Path]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in mask_paths:
        mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        values, value_counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, value_counts):
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return counts


def _edge_counts(mask_paths: list[Path]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in mask_paths:
        mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        values, value_counts = np.unique(edge, return_counts=True)
        for value, count in zip(values, value_counts):
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return counts


def infer_mask_encoding(root: str | Path, split: str = "train", max_masks: int = 20) -> MaskEncoding:
    mask_dir = Path(root) / split / "gts"
    mask_paths = sorted(mask_dir.glob("*.bmp"))[:max_masks]
    if not mask_paths and split != "val":
        mask_dir = Path(root) / "val" / "gts"
        mask_paths = sorted(mask_dir.glob("*.bmp"))[:max_masks]
    if not mask_paths:
        return MaskEncoding(background=255, cup=0)

    counts = _mask_counts(mask_paths)
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 3:
        raise ValueError(f"Expected at least 3 mask values under {mask_dir}, found {ranked}")

    edge_ranked = sorted(_edge_counts(mask_paths).items(), key=lambda item: item[1], reverse=True)
    background = edge_ranked[0][0]
    foreground = [(value, count) for value, count in ranked if value != background]
    if len(foreground) < 2:
        raise ValueError(f"Could not infer disc/cup labels under {mask_dir}; counts={ranked}")
    disc_rim = foreground[0][0]
    cup = foreground[-1][0]
    return MaskEncoding(background=background, disc_rim=disc_rim, cup=cup)


def summarize_mask_mapping(
    root: str | Path,
    split: str,
    encoding: MaskEncoding,
    max_masks: int = 20,
) -> dict[str, float | int]:
    mask_paths = sorted((Path(root) / split / "gts").glob("*.bmp"))[:max_masks]
    counts = {"background": 0, "disc_rim": 0, "cup": 0, "unknown": 0}
    total = 0
    mapping = {
        encoding.background: "background",
        encoding.disc_rim: "disc_rim",
        encoding.cup: "cup",
    }
    for path in mask_paths:
        mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        values, value_counts = np.unique(mask, return_counts=True)
        total += int(mask.size)
        for value, count in zip(values, value_counts):
            counts[mapping.get(int(value), "unknown")] += int(count)

    if total == 0:
        return {"num_masks": 0, "background_ratio": 0.0, "disc_rim_ratio": 0.0, "cup_ratio": 0.0}
    return {
        "num_masks": len(mask_paths),
        "background_ratio": counts["background"] / total,
        "disc_rim_ratio": counts["disc_rim"] / total,
        "cup_ratio": counts["cup"] / total,
        "foreground_ratio": (counts["disc_rim"] + counts["cup"]) / total,
        "unknown_ratio": counts["unknown"] / total,
    }


def validate_mask_mapping(root: str | Path, split: str, encoding: MaskEncoding) -> None:
    summary = summarize_mask_mapping(root, split, encoding)
    if summary["num_masks"] == 0:
        return
    problems = []
    if summary["unknown_ratio"] > 0:
        problems.append(f"unknown_ratio={summary['unknown_ratio']:.4f}")
    if summary["background_ratio"] < 0.7:
        problems.append(f"background_ratio={summary['background_ratio']:.4f}")
    if not 0 < summary["foreground_ratio"] < 0.3:
        problems.append(f"foreground_ratio={summary['foreground_ratio']:.4f}")
    if summary["disc_rim_ratio"] <= 0 or summary["cup_ratio"] <= 0:
        problems.append(
            f"disc_rim_ratio={summary['disc_rim_ratio']:.4f}, cup_ratio={summary['cup_ratio']:.4f}"
        )
    if problems:
        raise ValueError(
            f"Suspicious REFUGE mask mapping for split={split}: {encoding}; "
            f"{', '.join(problems)}. Check raw mask encoding before training."
        )


class REFUGEDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.images_dir = self.root / split / "Images"
        self.masks_dir = self.root / split / "gts"
        self.has_masks = self.masks_dir.exists()
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        self.mask_encoding = infer_mask_encoding(self.root, split) if self.has_masks else infer_mask_encoding(self.root)
        if self.has_masks:
            validate_mask_mapping(self.root, split, self.mask_encoding)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        do_hflip = self.augment and np.random.rand() < 0.5
        do_vflip = self.augment and np.random.rand() < 0.5
        sample = {
            "image": self._load_image(image_path, do_hflip=do_hflip, do_vflip=do_vflip),
            "id": image_path.stem,
        }
        if self.has_masks:
            mask_path = self.masks_dir / f"{image_path.stem}.bmp"
            sample["mask"] = self._load_mask(mask_path, do_hflip=do_hflip, do_vflip=do_vflip)
        return sample

    def _load_image(self, path: Path, do_hflip: bool, do_vflip: bool) -> torch.Tensor:
        image = Image.open(path)
        image.load()
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        if do_hflip:
            image = TF.hflip(image)
        if do_vflip:
            image = TF.vflip(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _load_mask(self, path: Path, do_hflip: bool, do_vflip: bool) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        mapped = np.zeros_like(mask, dtype=np.int64)
        for raw_value, class_id in self.mask_encoding.raw_to_class.items():
            mapped[mask == raw_value] = class_id
        if do_hflip:
            mapped = np.fliplr(mapped).copy()
        if do_vflip:
            mapped = np.flipud(mapped).copy()
        return torch.from_numpy(mapped)


def build_dataloaders(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = REFUGEDataset(
        root=cfg.data_root,
        split="train",
        image_size=cfg.image_size,
        augment=True,
    )
    val_dataset = REFUGEDataset(
        root=cfg.data_root,
        split="val",
        image_size=cfg.image_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
