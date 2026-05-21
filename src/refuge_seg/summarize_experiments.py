from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    ("CE", "outputs/checkpoints/task2_refuge_loss_ce"),
    ("Dice", "outputs/checkpoints/task2_refuge_loss_dice"),
    ("Dice+CE", "outputs/checkpoints/task2_refuge_baseline"),
    ("lr_1e-4", "outputs/checkpoints/task2_refuge_lr_1e4"),
    ("batch2", "outputs/checkpoints/task2_refuge_batch2"),
    ("step_scheduler", "outputs/checkpoints/task2_refuge_scheduler_step"),
    ("topology", "outputs/checkpoints/task2_refuge_topology"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "待补充"
    return str(value)


def build_table(experiments: list[tuple[str, Path]]) -> str:
    lines = [
        "| 实验 | loss | lr | batch | scheduler | avg epoch sec | mean dice | mean iou | cup outside disc |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, root in experiments:
        metrics = load_json(root / "eval_val" / "metrics.json")
        summary = load_json(root / "training_summary.json")
        cfg = summary.get("config", {})
        loss_cfg = cfg.get("loss", {})
        train_cfg = cfg.get("train", {})
        diag = metrics.get("diagnosis_summary", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(loss_cfg.get("name")),
                    fmt(train_cfg.get("lr")),
                    fmt(train_cfg.get("batch_size")),
                    fmt(train_cfg.get("scheduler")),
                    fmt(summary.get("avg_epoch_time_sec")),
                    fmt(metrics.get("mean_dice")),
                    fmt(metrics.get("mean_iou")),
                    fmt(diag.get("cup_outside_disc_cases")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="reports/task2_experiment_summary.md")
    args = parser.parse_args()

    experiments = [(name, Path(path)) for name, path in DEFAULT_EXPERIMENTS]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Task 2 实验结果汇总\n\n"
        "训练和评估完成后自动生成。若某项显示 `待补充`，说明对应实验产物尚不存在。\n\n"
        + build_table(experiments)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
