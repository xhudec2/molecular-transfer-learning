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
- [QM7 (DeepChem)](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv) [1][2]
- [Lipophilicity (DeepChem)](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv) [1][2]
- PubChem AID 504329 SD/DR data is retrieved via the [mf-pcba repository](https://github.com/davidbuterez/mf-pcba/) [3]

[1] - 10.48550/arXiv.1703.00564 
[2] - https://deepchem.readthedocs.io/en/latest/
[3] - 10.1021/acs.jcim.2c01569

**Note**: This step is time consuming and takes around 40 minutes to run. It is possible to run ```./get_datasets_simple.sh``` to skip downloading the MF-PCBA dataset, or unzip `data_splits.zip`.

## Repository structure

    molecular-transfer-learning
    ├── get_datasets.sh
    ├── get_datasets_simple.sh
    ├── pyproject.toml
    ├── README.md
    ├── data/                      # Data path 
    ├── notebooks/                 # Notebooks for faster experimentaion
    ├── report_data_scripts/       # Report data generation scripts       
    │   ├── plots.py
    │   └── tabular_data.py 
    ├── preprocessing/             # Data preprocessing scripts
    │   ├── clean.py
    │   └── splits.py
    ├── results/                   # All experimental results
    │   ├── dr/                    
    │   ├── lipo/                  
    │   ├── qm7/                   
    │   ├── optuna/                
    │   ├── tables/                
    │   └── README.md              
    └── vgae_model/
        ├── __init__.py
        ├── __main__.py
        ├── data/                  # Data loading and preprocessing
        │   ├── datamodule.py
        │   ├── dataset.py
        │   └── transforms.py
        ├── modelling/             # Model architecture
        │   ├── backbone.py
        │   ├── gcn.py
        │   ├── regressor.py
        │   ├── set_transformer.py
        │   └── vgaemodule.py
        └── utils/                 # Training utilities
            ├── hyperparams.py
            └── trainer.py

## Experiments (how to run)

The training/evaluation of the VGAE model is run via:

    uv run python -m vgae_model --help

Arguments:
- `--dataset`: sd, dr, lipo, qm7
- `--task`: pretrain, finetune, test
- `--experiment-name`: name of the output folder (results will be saved to `results/{dataset}/{model_type}/{experiment_name}`)
- `--ckpt`: path to a checkpoint file (required for `test`, used in `finetune` for transfer learning)
- `--lr`: learning rate (optional)
- `--seed` random seed (optional)

### Pretrain (example: SD)

    uv run python -m vgae_model --dataset sd --task pretrain --experiment-name sd_pretrain_all

### Hyperparameter optimization (example: DR from SD checkpoint)
    
    uv run python -m vgae_model --dataset dr --task hpopt --experiment-name dr_ft_hpopt --ckpt path/to/sd/checkpoint.ckpt

### Finetune (example: DR from SD checkpoint)

    uv run python -m vgae_model --dataset dr --task finetune --experiment-name dr_ft --ckpt path/to/sd/checkpoint.ckpt

### Test set evaluation (example: DR)

    uv run python -m vgae_model --dataset dr --task test --experiment-name best_dr_test --ckpt path/to/checkpoint.ckpt

## Results

All experiment outputs are saved to the `results/` directory, organized by dataset and model type:

```
results/
├── dr/           # Dose Response dataset results
├── lipo/         # Lipophilicity dataset results
├── qm7/          # QM7 dataset results
├── optuna/       # Hyperparameter optimization trials
├── tables/       # Summary tables and statistics
├── figures/      # Dataset visualization
└── README.md     # Detailed results documentation
```

For each dataset, results are organized as:

- `{dataset}/non-pretrained/` — Models trained from scratch
- `{dataset}/pretrained/` — Models using transfer learning
- `{dataset}/rf/` — Random Forest baseline

Each model contains:
- `hparams.yaml` — Hyperparameters used for training
- `metrics.csv` — Test set metrics (MAE, RMSE, R2)
- Checkpoints saved during training

See [results/README.md](results/README.md) for detailed documentation on interpreting the results.

## Models

We do not provide any trained models in the dataset as they are large and take up a lot of space.

## AI usage
During the development phase we used GitHub Copilot and other LLMs to speed up the process. We also used these tools for writing documentation and always checked the outputs.
