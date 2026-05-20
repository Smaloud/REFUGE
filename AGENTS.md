# Repository Guidelines

## Project Structure & Module Organization
This repository focuses on `Task 2` of the course project: REFUGE optic disc/cup segmentation. Raw data lives under `REFUGE/{train,val,test}`. Source code is under `src/refuge_seg/`, split into `datasets/`, `models/`, `losses/`, and `utils/`. Training configs live in `configs/`, runnable shell entrypoints in `scripts/`, and experiment outputs in `outputs/`. Use `reports/task2_answer_template.md` as the final write-up template.

## Build, Test, and Development Commands
Run all commands from the repository root with `PYTHONPATH=src`.

```bash
bash scripts/plot_task2_dataset.sh
bash scripts/train_task2_baseline.sh
bash scripts/train_task2_topology.sh
bash scripts/eval_task2.sh
bash scripts/eval_task2_topology.sh
bash scripts/plot_task2_results.sh
bash scripts/plot_task2_optional.sh
python3 -m refuge_seg.predict --config configs/task2_refuge_baseline.yaml --checkpoint outputs/checkpoints/task2_refuge_baseline/best_model.pt --input_dir REFUGE/test/Images --output_dir outputs/predictions/test --postprocess
```

These commands generate dataset figures, train the baseline and topology-aware models, save validation metrics and qualitative outputs, summarize curves/metrics for slides, and export test-set masks.

## Coding Style & Naming Conventions
Use Python 3 with 4-space indentation and type hints where practical. Keep module names lowercase with underscores, class names in `CamelCase`, and config files named by task and experiment, for example `task2_refuge_topology.yaml`. Prefer explicit functions over notebook-only logic so experiments are reproducible.

## Testing Guidelines
There is no formal test suite yet; perform static checks before training:

```bash
PYTHONPATH=src python3 -m py_compile $(find src -name '*.py')
```

For behavioral checks, run a short validation pass and inspect `outputs/checkpoints/.../best_prediction.png`. Report metrics with `Dice_disc`, `Dice_cup`, `IoU_disc`, `IoU_cup`, and `mean_dice`.

## Commit & Pull Request Guidelines
This directory is not currently a Git repository, so adopt a simple imperative commit style if version control is initialized later, for example `add topology-aware refuge training pipeline`. Pull requests should include the goal of the experiment, config changes, main metrics, and representative prediction figures, especially before/after post-processing or topology constraints.

## Security & Configuration Tips
Do not edit raw data in `REFUGE/`. Write generated masks, plots, and checkpoints only under `outputs/`. On shared servers, set `MPLCONFIGDIR=/tmp/mplconfig` to avoid Matplotlib cache permission issues.
