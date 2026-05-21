from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from refuge_seg.datasets import DatasetConfig, build_dataloaders
from refuge_seg.losses import build_loss
from refuge_seg.models import AttentionUNet, UNet
from refuge_seg.utils.metrics import SegmentationMeter
from refuge_seg.utils.visualization import save_prediction_grid, save_training_curves


def build_model(name: str, num_classes: int, base_channels: int) -> torch.nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(num_classes=num_classes, base_channels=base_channels)
    if name == "attention_unet":
        return AttentionUNet(num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"Unsupported model: {name}")


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    if name.lower() == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(name: str, optimizer, epochs: int):
    if name.lower() == "none":
        return None
    if name.lower() == "step":
        return StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    if name.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    meter = SegmentationMeter()
    sample_triplet = None
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * images.size(0)
        meter.update(preds.cpu(), masks.cpu())
        if sample_triplet is None:
            sample_triplet = (images[0].cpu(), masks[0].cpu(), preds[0].cpu())
    metrics = meter.compute()
    metrics["val_loss"] = total_loss / len(loader.dataset)
    return metrics, sample_triplet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task2_refuge_baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = DatasetConfig(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    train_loader, val_loader = build_dataloaders(dataset_cfg)
    cfg["data"]["mask_encoding"] = train_loader.dataset.mask_encoding.class_to_raw

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    criterion = build_loss(cfg["loss"]["name"], cfg["loss"].get("lambda_topology", 0.3))
    optimizer = build_optimizer(
        cfg["train"]["optimizer"],
        model.parameters(),
        cfg["train"]["lr"],
        cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(cfg["train"]["scheduler"], optimizer, cfg["train"]["epochs"])

    history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}
    best_score = -1.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, sample_triplet = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_mean_dice"].append(val_metrics["mean_dice"])

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        print(json.dumps(epoch_log, ensure_ascii=False))

        if val_metrics["mean_dice"] > best_score:
            best_score = val_metrics["mean_dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )
            if sample_triplet is not None:
                save_prediction_grid(*sample_triplet, output_dir / "best_prediction.png")

    save_training_curves(history, output_dir / "training_curves.png")
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
