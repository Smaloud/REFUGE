import numpy as np
import torch
from PIL import Image

from refuge_seg.datasets.refuge_dataset import REFUGEDataset, infer_mask_encoding
from refuge_seg.utils.metrics import SegmentationMeter
from refuge_seg.utils.postprocess import diagnose_prediction, postprocess_prediction


def test_disc_metrics_include_cup_region() -> None:
    target = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])
    pred = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])

    meter = SegmentationMeter()
    meter.update(pred, target)
    metrics = meter.compute()

    assert metrics["dice_disc"] == 1.0
    assert metrics["dice_cup"] == 1.0


def test_empty_prediction_does_not_get_high_disc_score() -> None:
    target = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])
    pred = torch.zeros_like(target)

    meter = SegmentationMeter()
    meter.update(pred, target)
    metrics = meter.compute()

    assert metrics["dice_disc"] < 1e-5
    assert metrics["dice_cup"] < 1e-5


def test_postprocess_keeps_cup_inside_complete_disc() -> None:
    pred = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    processed = postprocess_prediction(pred)
    diagnosis = diagnose_prediction(processed)

    assert processed[2, 2] == 2
    assert diagnosis["disc_components"] == 1
    assert diagnosis["cup_components"] == 1
    assert not diagnosis["cup_outside_disc"]


def test_mask_encoding_infers_255_background(tmp_path) -> None:
    mask_dir = tmp_path / "train" / "gts"
    image_dir = tmp_path / "train" / "Images"
    mask_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image_dir / "case1.jpg")
    mask = np.array(
        [
            [255, 255, 255, 255],
            [255, 128, 128, 255],
            [255, 128, 0, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask).save(mask_dir / "case1.bmp")

    encoding = infer_mask_encoding(tmp_path)
    dataset = REFUGEDataset(tmp_path, "train", image_size=4)
    sample = dataset[0]["mask"]

    assert encoding.background == 255
    assert encoding.cup == 0
    assert sample[2, 2].item() == 2


def test_mask_encoding_infers_0_background(tmp_path) -> None:
    mask_dir = tmp_path / "train" / "gts"
    image_dir = tmp_path / "train" / "Images"
    mask_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image_dir / "case1.jpg")
    mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 128, 128, 0],
            [0, 128, 255, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask).save(mask_dir / "case1.bmp")

    encoding = infer_mask_encoding(tmp_path)
    dataset = REFUGEDataset(tmp_path, "train", image_size=4)
    sample = dataset[0]["mask"]

    assert encoding.background == 0
    assert encoding.cup == 255
    assert sample[2, 2].item() == 2


def test_mask_encoding_infers_by_frequency_when_labels_differ(tmp_path) -> None:
    mask_dir = tmp_path / "train" / "gts"
    image_dir = tmp_path / "train" / "Images"
    mask_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image_dir / "case1.jpg")
    mask = np.array(
        [
            [10, 10, 10, 10],
            [10, 20, 20, 10],
            [10, 30, 30, 10],
            [10, 10, 10, 10],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask).save(mask_dir / "case1.bmp")

    encoding = infer_mask_encoding(tmp_path)

    assert encoding.background == 10
    assert encoding.disc_rim == 20
    assert encoding.cup == 30
