from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from refuge_seg.models import AttentionUNet, UNet
from refuge_seg.utils.postprocess import postprocess_prediction


def build_model(name: str, num_classes: int, base_channels: int):
    if name == "unet":
        return UNet(num_classes=num_classes, base_channels=base_channels)
    if name == "attention_unet":
        return AttentionUNet(num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"Unsupported model: {name}")


def preprocess(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task2_refuge_baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(Path(args.input_dir).glob("*.jpg")):
        image = preprocess(image_path, cfg["data"]["image_size"]).to(device)
        pred = torch.argmax(model(image), dim=1)[0].cpu().numpy().astype(np.uint8)
        if args.postprocess:
            pred = postprocess_prediction(pred)
        save_mask = np.zeros_like(pred, dtype=np.uint8)
        save_mask[pred == 1] = 128
        save_mask[pred == 2] = 255
        Image.fromarray(save_mask).save(output_dir / f"{image_path.stem}.bmp")


if __name__ == "__main__":
    main()

