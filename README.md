# Palsy Detection

Training code for facial palsy grade classification with config-driven experiments.

## Project Structure

- `src/preprocessing/`: manifest builders, dataset classes, transforms
- `src/models/`: model definitions
- `src/training/`: training engine, experiment config loader, run scripts
- `configs/experiments.toml`: set-and-forget experiment definitions
- `data/manifests/`: parquet/csv training manifests
- `data/runs/`: model checkpoints and training histories

## Run a Single Baseline

Multi-pose baseline:

```bash
python -m src.training.train_baseline
```

Single-image baseline:

```bash
python -m src.training.train_single
```

## Experiment Config Usage

Experiments are defined in `configs/experiments.toml`:

- `[defaults]`: shared settings for all runs
- `[[experiments]]`: one entry per run

Required fields per experiment:

- `run_name`
- `task_type`: `multi_pose` or `single_image`
- `manifest_path`

Optional useful fields:

- `model_name`: `efficientnet_b0_multi_pose` or `efficientnet_b0_single_image`
- `learning_rate`, `dropout`, `batch_size`, `num_epochs`, `use_class_weights`, etc.
- `seed`: reproducibility
- `repeats`: run the same experiment multiple times (auto-appends suffix `_r01`, `_r02`, ...)

### Minimal example

```bash
cat > configs/experiments.toml <<'EOF'
[defaults]
output_dir = "../data/runs"
num_epochs = 10
batch_size = 4
num_workers = 0
seed = 123

[[experiments]]
run_name = "baseline_multi_pose"
task_type = "multi_pose"
model_name = "efficientnet_b0_multi_pose"
manifest_path = "../data/manifests/training_manifest_images_only.parquet"
learning_rate = 1e-4
dropout = 0.3
use_class_weights = true
EOF
```

### Sweep example (different settings + models)

```bash
cat > configs/experiments.toml <<'EOF'
[defaults]
output_dir = "../data/runs"
num_epochs = 15
batch_size = 4
num_workers = 0
pretrained = true
freeze_backbone = true

[[experiments]]
run_name = "mp_lr1e4"
task_type = "multi_pose"
model_name = "efficientnet_b0_multi_pose"
manifest_path = "../data/manifests/training_manifest_images_only.parquet"
learning_rate = 1e-4
dropout = 0.3
use_class_weights = true

[[experiments]]
run_name = "mp_lr5e5"
task_type = "multi_pose"
model_name = "efficientnet_b0_multi_pose"
manifest_path = "../data/manifests/training_manifest_images_only.parquet"
learning_rate = 5e-5
dropout = 0.3
use_class_weights = true

[[experiments]]
run_name = "si_express"
task_type = "single_image"
model_name = "efficientnet_b0_single_image"
manifest_path = "../data/manifests/training_manifest_single_images.parquet"
transform_name = "express"
learning_rate = 1e-5
dropout = 0.3
use_class_weights = false
EOF
```

### Repeat example (same experiment multiple times)

```bash
cat > configs/experiments.toml <<'EOF'
[defaults]
output_dir = "../data/runs"
num_epochs = 10
batch_size = 4
seed = 100

[[experiments]]
run_name = "mp_repeat"
task_type = "multi_pose"
model_name = "efficientnet_b0_multi_pose"
manifest_path = "../data/manifests/training_manifest_images_only.parquet"
learning_rate = 1e-4
repeats = 3
EOF
```

This creates runs:

- `mp_repeat_r01` (seed 100)
- `mp_repeat_r02` (seed 101)
- `mp_repeat_r03` (seed 102)

## Run Commands

Run every experiment in file:

```bash
python -m src.training.run_experiments --config configs/experiments.toml
```

Run only specific names:

```bash
python -m src.training.run_experiments --config configs/experiments.toml --only mp_lr1e4 si_express
```

Stop on first failure:

```bash
python -m src.training.run_experiments --config configs/experiments.toml --stop-on-error
```

## Set-And-Forget Pattern

Recommended workflow:

1. Create one TOML file per batch, for example `configs/nightly_2026_03_25.toml`.
2. Set all runs you want inside that file.
3. Start once:
   `python -m src.training.run_experiments --config configs/nightly_2026_03_25.toml`
4. Leave it running; every run writes its own checkpoint/history.

## Outputs and Naming

Each run writes:

- Checkpoint: `data/runs/<run_name>.pt`
- History: `data/runs/<run_name>_history.json`

The history file includes the effective config and epoch metrics (`loss`, `acc`, optional `mae`, `lr`).
