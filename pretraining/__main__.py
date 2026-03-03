from lightning.pytorch.loggers import MLFlowLogger
import lightning as pl
from pretraining.modelling import VGAEModule
from pretraining.data import GeometricDataModule
from random import randint


def train() -> None:
    SEED = randint(0, 2**32 - 1)
    pl.seed_everything(SEED)

    hyperparams = {
        "batch_size": 512,
        "lr": 0.00005,
        "max_atom_num": 53,
        "seed": SEED,
        "max_epochs": 150,
        "ckpt": None,
        "experiment_name": ...
    }

    dm = GeometricDataModule(
        batch_size=hyperparams["batch_size"],
        seed=hyperparams["seed"],
        max_atom_num=hyperparams["max_atom_num"],
        train_path="./data/sd_splits/sd_train.csv",
        separate_valid_path="./data/sd_splits/sd_val.csv",
        label_column_name="SD",
        num_cores=(7, 7, 7),
        use_standard_scaler=True,
    )

    if hyperparams["ckpt"] is None:
        model = VGAEModule(
            num_features=hyperparams["max_atom_num"] + 27,
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
        )
    else:
        model = VGAEModule.load_from_checkpoint(
            hyperparams["ckpt"],
            num_features=hyperparams["max_atom_num"] + 27,
        )

    mlflow_logger = MLFlowLogger(
        experiment_name=hyperparams["experiment_name"],
        tracking_uri="databricks",
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        logger=mlflow_logger,
        deterministic=True,
        enable_checkpointing=True,
    )
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm, ckpt_path=hyperparams["ckpt"])


if __name__ == "__main__":
    train()
