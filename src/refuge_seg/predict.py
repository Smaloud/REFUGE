from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torchvision.transforms import functional as TF

from refuge_seg.datasets.refuge_dataset import MaskEncoding, _normalize_token, infer_mask_encoding
from refuge_seg.models import AttentionUNet, UNet
from refuge_seg.utils.postprocess import postprocess_prediction

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def load_mask_encoding(cfg: dict, checkpoint: dict, input_dir: Path) -> MaskEncoding:
    saved = checkpoint.get("config", {}).get("data", {}).get("mask_encoding")
    configured = cfg.get("data", {}).get("mask_encoding")
    encoding = saved or configured
    if encoding:
        normalized = {int(key): _normalize_token(value) for key, value in encoding.items()}
        return MaskEncoding(
            background=normalized[0],
            disc_rim=normalized[1],
            cup=normalized[2],
        )

    data_root = cfg.get("data", {}).get("root")
    if data_root:
        return infer_mask_encoding(data_root)

    parent = input_dir.parent.parent
    if (parent / "train" / "gts").exists() or (parent / "val" / "gts").exists():
        return infer_mask_encoding(parent)
    return MaskEncoding(background=255, disc_rim=128, cup=0)


def raw_mask_value(value) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value)


def is_grayscale_token(value) -> bool:
    if not isinstance(value, (tuple, list)):
        return True
    return len(value) >= 3 and int(value[0]) == int(value[1]) == int(value[2])


def raw_rgb_value(value) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value[:3])
    gray = int(value)
    return gray, gray, gray


def render_prediction_mask(pred: np.ndarray, encoding: MaskEncoding) -> np.ndarray:
    raw_values = encoding.class_to_raw
    if all(is_grayscale_token(value) for value in raw_values.values()):
        save_mask = np.full_like(pred, raw_mask_value(raw_values[0]), dtype=np.uint8)
        save_mask[pred == 1] = raw_mask_value(raw_values[1])
        save_mask[pred == 2] = raw_mask_value(raw_values[2])
        return save_mask

    save_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    save_mask[:, :] = raw_rgb_value(raw_values[0])
    save_mask[pred == 1] = raw_rgb_value(raw_values[1])
    save_mask[pred == 2] = raw_rgb_value(raw_values[2])
    return save_mask


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
    mask_encoding = load_mask_encoding(cfg, checkpoint, Path(args.input_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for image_path in sorted(Path(args.input_dir).glob("*.jpg")):
        image = preprocess(image_path, cfg["data"]["image_size"]).to(device)
        pred = torch.argmax(model(image), dim=1)[0].cpu().numpy().astype(np.uint8)
        if args.postprocess:
            pred = postprocess_prediction(pred)
        values, counts = np.unique(pred, return_counts=True)
        summary[image_path.stem] = {str(int(value)): int(count) for value, count in zip(values, counts)}
        save_mask = render_prediction_mask(pred, mask_encoding)
        Image.fromarray(save_mask).save(output_dir / f"{image_path.stem}.bmp")

    unique_distributions = {tuple(sorted(item.items())) for item in summary.values()}
    report = {
        "mask_encoding": mask_encoding.class_to_raw,
        "num_images": len(summary),
        "num_unique_class_distributions": len(unique_distributions),
        "class_distribution_examples": dict(list(summary.items())[:10]),
    }
    with open(output_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
