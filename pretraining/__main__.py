from lightning.pytorch.loggers import MLFlowLogger
import lightning as pl
from pretraining.modelling import VGAEModule
from pretraining.data import GeometricDataModule
import mlflow
from random import randint


def train() -> None:
    SEED = randint(0, 2**32 - 1)
    URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(URI)
    pl.seed_everything(SEED)

    hyperparams = {
        "batch_size": 512,
        "lr": 0.00005,
        "max_atom_num": 53,
        "seed": SEED,
        "max_epochs": 200,
    }

    dm = GeometricDataModule(
        batch_size=hyperparams["batch_size"],
        seed=hyperparams["seed"],
        max_atom_num=hyperparams["max_atom_num"],
        train_path="./data/SD_debug_subset_train.csv",
        separate_valid_path="./data/SD_debug_subset_val.csv",
        id_column="CID",
        label_column_name="SD",
        num_cores=(4, 0, 4),
        use_standard_scaler=True,
    )

    model = VGAEModule(
        num_features=hyperparams["max_atom_num"] + 27,
        lr=hyperparams["lr"],
        batch_size=hyperparams["batch_size"],
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=hyperparams["max_epochs"],
        logger=(MLFlowLogger("Pretraining", tracking_uri=URI)),
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
