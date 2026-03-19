# Cross Domain Molecular Transfer Learning

This repository trains and evaluates a graph-based model for molecular property prediction with transfer learning across datasets.

## Installation

This repository uses uv for dependency management.

Install uv:

    pip install uv

Install dependencies:

    uv sync

## Data

To download the datasets used in this repository and preprocess them run:

    ./get_datasets.sh

This creates train / val / test splits:
- data/sd_splits
- data/dr_splits
- data/lipo_splits
- data/qm7_splits

Data sources used by the download script:
- [QM7 (DeepChem)](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv)
- [Lipophilicity (DeepChem)](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv)
- PubChem AID 504329 SD/DR data is retrieved via the [mf-pcba repository](https://github.com/davidbuterez/mf-pcba/)

**Note**: This step is time consuming and takes around 40 minutes to run.

## Repository structure

    molecular-transfer-learning
    ├── download_data.sh
    ├── pyproject.toml
    ├── README.md
    ├── uv.lock
    ├── checkpoints
    ├── data
    │   ├── dr_splits
    │   │   ├── dr_test.csv
    │   │   ├── dr_train.csv
    │   │   └── dr_val.csv
    │   ├── lipo_splits
    │   │   ├── lipo_test.csv
    │   │   ├── lipo_train.csv
    │   │   └── lipo_val.csv
    │   ├── qm7_splits
    │   │   ├── qm7_test.csv
    │   │   ├── qm7_train.csv
    │   │   └── qm7_val.csv
    │   └── sd_splits
    │       ├── sd_all.csv
    │       ├── sd_test.csv
    │       ├── sd_train.csv
    │       └── sd_val.csv
    ├── lightning_logs
    ├── notebooks
    └── vgae_model
        ├── __init__.py
        ├── __main__.py
        ├── data
        │   ├── __init__.py
        │   ├── datamodule.py
        │   ├── dataset.py
        │   └── transforms.py
        ├── modelling
        │   ├── __init__.py
        │   ├── backbone.py
        │   ├── gcn.py
        │   ├── regressor.py
        │   ├── set_transformer.py
        │   └── vgaemodule.py
        └── utils
            ├── __init__.py
            ├── hyperparams.py
            └── trainer.py

## Experiments (how to run)

The training/evaluation of the VGAE model is run via:

    uv run python -m vgae_model --help

Arguments:
- `--dataset`: sd, dr, lipo, qm7
- `--task`: pretrain, finetune, test
- `--experiment-name`: name of the output folder under lightning_logs
- `--ckpt`: path to a checkpoint file (required for `test`, used in `finetune` for transfer learning)
- `--lr`: learning rate (optional)

### Pretrain (example: SD)

    uv run python -m vgae_model --dataset sd --task pretrain --experiment-name sd_pretrain_all

### Finetune (example: DR from SD checkpoint)

    uv run python -m vgae_model --dataset dr --task finetune --experiment-name dr_ft --ckpt lightning_logs/sd_pretrain_all/checkpoints/epoch=199-step=124400.ckpt

### Test set evaluation (example: DR)

    uv run python -m vgae_model --dataset dr --task test --experiment-name best_dr_test --ckpt lightning_logs/version_20/checkpoints/epoch=14-step=330.ckpt

## Expected outputs

For each experiment name NAME, outputs are written to:

- lightning_logs/NAME/hparams.yaml
  - run configuration (dataset, lr, seed, paths, etc.)
- lightning_logs/NAME/metrics.csv
  - logged metrics (train/val/test depending on task)
- lightning_logs/NAME/checkpoints
  - saved checkpoints

Example existing logs you can inspect:
- lightning_logs/sd_pretrain_all
- lightning_logs/best_dr_test
- lightning_logs/qm_7_test


## AI usage
During the development phase we used GitHub Copilot and other LLMs to speed up the process. We also used these tools for writing documentation and always checked the outputs.
