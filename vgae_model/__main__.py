from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning as pl
from vgae_model.modelling import VGAEModule
from vgae_model.data import GeometricDataModule
from random import randint


def train() -> None:
    SEED = randint(0, 2**32 - 1)
    pl.seed_everything(SEED)

    hyperparams = {
        "batch_size": 32,
        "lr": 0.00005,
        "max_atom_num": 53,
        "seed": SEED,
        "max_epochs": 200,
        "num_pretrain_epochs": 150,
        "smiles_column": "neut-smiles",
        "train_path": "./data/dr_splits/dr_train.csv",
        "separate_valid_path": "./data/dr_splits/dr_val.csv",
        "label_column_name": "DR",
        "ckpt": None,
    }

    dm = GeometricDataModule(
        batch_size=hyperparams["batch_size"],
        seed=hyperparams["seed"],
        max_atom_num=hyperparams["max_atom_num"],
        train_path=hyperparams["train_path"],
        separate_valid_path=hyperparams["separate_valid_path"],
        label_column_name=hyperparams["label_column_name"],
        smiles_column=hyperparams["smiles_column"],
        num_cores=(0, 0, 0),
        use_standard_scaler=True,
    )

    if hyperparams["ckpt"] is None:
        model = VGAEModule(
            num_features=hyperparams["max_atom_num"] + 27,
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            num_pretrain_epochs=hyperparams["num_pretrain_epochs"],
        )
    else:
        model = VGAEModule.load_from_checkpoint(
            hyperparams["ckpt"],
            num_features=hyperparams["max_atom_num"] + 27,
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            num_pretrain_epochs=hyperparams["num_pretrain_epochs"],
        )

    logger = CSVLogger(save_dir=".")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mae",  # maybe something else?
        mode="min",
        save_top_k=1,
    )

    stopper = EarlyStopping(
        monitor="val/mae",
        mode="min",
        patience=20,
    )

    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, stopper],
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=50,
    )
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm, ckpt_path=hyperparams["ckpt"])


if __name__ == "__main__":
    train()
