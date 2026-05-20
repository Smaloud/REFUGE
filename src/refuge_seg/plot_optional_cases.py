from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from refuge_seg.datasets.refuge_dataset import REFUGEDataset
from refuge_seg.models import AttentionUNet, UNet
from refuge_seg.utils.postprocess import diagnose_prediction, postprocess_prediction
from refuge_seg.utils.visualization import colorize_mask, denormalize


def build_model(name: str, num_classes: int, base_channels: int):
    if name == "unet":
        return UNet(num_classes=num_classes, base_channels=base_channels)
    if name == "attention_unet":
        return AttentionUNet(num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"Unsupported model: {name}")


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return cfg, model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--topology-config", type=str, required=True)
    parser.add_argument("--topology-ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/figures/task2_optional_cases.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_cfg, baseline_model = load_model(args.baseline_config, args.baseline_ckpt, device)
    topology_cfg, topology_model = load_model(args.topology_config, args.topology_ckpt, device)

    dataset = REFUGEDataset(
        root=baseline_cfg["data"]["root"],
        split=args.split,
        image_size=baseline_cfg["data"]["image_size"],
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    hard_cases = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"][0].cpu().numpy()

        baseline_pred = torch.argmax(baseline_model(image), dim=1)[0].cpu().numpy()
        topology_pred = torch.argmax(topology_model(image), dim=1)[0].cpu().numpy()
        baseline_post = postprocess_prediction(baseline_pred)

        diag = diagnose_prediction(baseline_pred)
        severity = (
            int(diag["cup_outside_disc"]) * 10
            + int(diag["disc_components"])
            + int(diag["cup_components"])
            + int(diag["disc_holes_pixels"] > 0)
            + int(diag["cup_holes_pixels"] > 0)
        )
        hard_cases.append(
            {
                "id": batch["id"][0],
                "image": batch["image"][0].cpu(),
                "target": target,
                "baseline": baseline_pred,
                "baseline_post": baseline_post,
                "topology": topology_pred,
                "severity": severity,
                "diag": diag,
            }
        )

    hard_cases.sort(key=lambda item: item["severity"], reverse=True)
    selected = hard_cases[: args.num_cases]

    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["原图", "真值", "基线预测", "后处理后", "拓扑约束预测"]
    for row, case in enumerate(selected):
        images = [
            denormalize(case["image"]),
            colorize_mask(case["target"]),
            colorize_mask(case["baseline"]),
            colorize_mask(case["baseline_post"]),
            colorize_mask(case["topology"]),
        ]
        for col, img in enumerate(images):
            axes[row, col].imshow(img)
            axes[row, col].set_title(titles[col] if row == 0 else "")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(case["id"], rotation=90, fontsize=9)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

