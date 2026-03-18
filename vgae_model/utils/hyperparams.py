from random import randint
from typing import Any


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
}

QM7 = {
    "batch_size": 32,
    "train_path": "./data/qm7_splits/qm7_train.csv",
    "separate_valid_path": "./data/qm7_splits/qm7_val.csv",
    "separate_test_path": "./data/qm7_splits/qm7_test.csv",
    "label_column_name": "u0_atom",
}

DR = {
    "batch_size": 32,
    "train_path": "./data/dr_splits/dr_train.csv",
    "separate_valid_path": "./data/dr_splits/dr_val.csv",
    "separate_test_path": "./data/dr_splits/dr_test.csv",
    "label_column_name": "DR",
}

SD = {
    "batch_size": 512,
    "train_path": "./data/SD.csv",
    "separate_valid_path": None,
    "separate_test_path": None,
    "label_column_name": "SD",
}


def get_hparams(
    dataset: str, task: str, experiment_name: None | str, ckpt: None | str, lr: float
) -> dict[str, Any]:
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
