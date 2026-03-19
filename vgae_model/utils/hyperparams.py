"""Hyperparameters and dataset paths.

`get_hparams` returns a dictionary used by `vgae_model.utils.trainer`
and logged by Lightning (to `hparams.yaml`) for reproducibility.
The seed is randomly generated per run by default, also logged for reproducibility.
"""

from random import randint
from typing import Any

# these values are kept as base and are not changed, as we did not try to change them during training.
# the values are the same as those described in https://www.nature.com/articles/s41467-024-45566-8#Sec20
BASE_CONFIG = {
    "max_atom_num": 53,
    "max_epochs": 200,
    "num_cores": (15, 15, 15),
    "smiles_column": "smiles",
    "ckpt": None,
}

LIPO = {
    "batch_size": 32,
    "train_path": "./data/lipo_splits/lipo_train.csv",
    "separate_valid_path": "./data/lipo_splits/lipo_val.csv",
    "separate_test_path": "./data/lipo_splits/lipo_test.csv",
    "label_column_name": "exp",
    "monitor_loss": "train/val_rmse",
}

QM7 = {
    "batch_size": 32,
    "train_path": "./data/qm7_splits/qm7_train.csv",
    "separate_valid_path": "./data/qm7_splits/qm7_val.csv",
    "separate_test_path": "./data/qm7_splits/qm7_test.csv",
    "label_column_name": "u0_atom",
    "monitor_loss": "train/val_rmse",
}

DR = {
    "batch_size": 32,
    "train_path": "./data/dr_splits/dr_train.csv",
    "separate_valid_path": "./data/dr_splits/dr_val.csv",
    "separate_test_path": "./data/dr_splits/dr_test.csv",
    "label_column_name": "DR",
    "monitor_loss": "train/val_rmse",
}

SD = {
    "batch_size": 512,
    "train_path": "./data/SD.csv",
    "separate_valid_path": None,
    "separate_test_path": None,
    "label_column_name": "SD",
    "monitor_loss": "train/total_loss",
}


def get_hparams(
    dataset: str, task: str, experiment_name: None | str, ckpt: None | str, lr: float
) -> dict[str, Any]:
    """Construct the hyperparameter dict for a given dataset/task.

    Args:
        dataset: One of `sd`, `dr`, `lipo`, `qm7`.
        task: One of `pretrain`, `finetune`, `test`, `hpopt`.
        experiment_name: Lightning log folder name under `lightning_logs/`.
        ckpt: Optional checkpoint for finetuning, not optional for testing.
        lr: Learning rate.

    Returns:
        A dictionary containing training settings and dataset CSV paths.
    """
    SEED = randint(0, 2**32 - 1)
    hyperparams = BASE_CONFIG.copy()
    hyperparams["seed"] = SEED
    hyperparams["task"] = task

    hyperparams["experiment_name"] = experiment_name
    hyperparams["ckpt"] = ckpt
    hyperparams["lr"] = lr

    match dataset:
        case "sd":
            hyperparams.update(SD.copy())
        case "dr":
            hyperparams.update(DR.copy())
        case "lipo":
            hyperparams.update(LIPO.copy())
        case "qm7":
            hyperparams.update(QM7.copy())

    return hyperparams
