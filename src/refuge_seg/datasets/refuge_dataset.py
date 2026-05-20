from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


MASK_TO_CLASS = {
    255: 0,  # background
    128: 1,  # optic disc
    0: 2,  # optic cup
}


@dataclass
class DatasetConfig:
    data_root: str
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 4


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
        image = Image.open(path).convert("RGB")
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
        for raw_value, class_id in MASK_TO_CLASS.items():
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
