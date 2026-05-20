from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from refuge_seg.datasets.refuge_dataset import REFUGEDataset
from refuge_seg.models import AttentionUNet, UNet
from refuge_seg.utils.metrics import SegmentationMeter
from refuge_seg.utils.postprocess import diagnose_prediction, postprocess_prediction
from refuge_seg.utils.visualization import save_prediction_grid


def build_model(name: str, num_classes: int, base_channels: int):
    if name == "unet":
        return UNet(num_classes=num_classes, base_channels=base_channels)
    if name == "attention_unet":
        return AttentionUNet(num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"Unsupported model: {name}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task2_refuge_baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = REFUGEDataset(
        root=cfg["data"]["root"],
        split=args.split,
        image_size=cfg["data"]["image_size"],
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"])

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    meter = SegmentationMeter()
    diagnosis = []
    output_dir = Path(cfg["output_dir"]) / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, batch in enumerate(tqdm(loader, desc="evaluate")):
        images = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()

        if args.postprocess:
            post_preds = []
            for pred in preds:
                post_preds.append(torch.from_numpy(postprocess_prediction(pred.numpy())))
            preds = torch.stack(post_preds)

        meter.update(preds, masks)
        diagnosis.append(diagnose_prediction(preds[0].numpy()))

        if index < 8:
            save_prediction_grid(
                batch["image"][0].cpu(),
                masks[0].cpu(),
                preds[0].cpu(),
                output_dir / f"{batch['id'][0]}.png",
            )

    summary = meter.compute()
    summary["diagnosis_examples"] = diagnosis[:20]
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

