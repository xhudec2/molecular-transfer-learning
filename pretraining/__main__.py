from lightning.pytorch.loggers import MLFlowLogger
import lightning as pl
from pretraining.modelling import VGAEModule
from pretraining.data import GeometricDataModule
import mlflow
from random import randint


def train() -> None:
    SEED = randint(0, 2**32 - 1)
    URI = "http://127.0.0.1:8080"
    mlflow.set_tracking_uri(URI)
    pl.seed_everything(SEED)

    dm = GeometricDataModule(
        32,
        42,
        35,
        train_path="./data/SD_debug_subset_train.csv",
        separate_valid_path="./data/SD_debug_subset_val.csv",
        id_column="CID",
        label_column_name="SD",
        num_cores=(4, 0, 4),
        use_standard_scaler=True,
    )
    model = VGAEModule(62)
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10,
        logger=(MLFlowLogger("Pretraining", tracking_uri=URI)),
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
