from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
import lightning as pl
from vgae_model.modelling import VGAEModule, VGAERegressionHead
from vgae_model.data import GeometricDataModule


def _get_datamodule(hyperparams) -> GeometricDataModule:
    return GeometricDataModule(
        batch_size=hyperparams["batch_size"],
        seed=hyperparams["seed"],
        max_atom_num=hyperparams["max_atom_num"],
        train_path=hyperparams["train_path"],
        separate_valid_path=hyperparams.get("separate_valid_path", None),
        separate_test_path=hyperparams.get("separate_test_path", None),
        label_column_name=hyperparams["label_column_name"],
        smiles_column=hyperparams["smiles_column"],
        num_cores=hyperparams["num_cores"],
        use_standard_scaler=True,
    )


def _get_model(hyperparams, scaler, testing=False):
    if hyperparams["ckpt"] is None:
        return VGAEModule(
            num_features=hyperparams["max_atom_num"] + 27,
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            num_pretrain_epochs=hyperparams["num_pretrain_epochs"],
            scaler=scaler,
        )

    model = VGAEModule.load_from_checkpoint(
        hyperparams["ckpt"],
        num_features=hyperparams["max_atom_num"] + 27,
        lr=hyperparams["lr"],
        batch_size=hyperparams["batch_size"],
        num_pretrain_epochs=hyperparams["num_pretrain_epochs"],
        scaler=scaler,
    )

    if not testing:
        model.regressor = VGAERegressionHead()
        for param in model.backbone.gnn_model.parameters():
            param.requires_grad = False

    return model


def pretrain(hyperparams) -> None:
    if hyperparams["ckpt"] is not None:
        raise ValueError("Pretraining does not expect a checkpoint parameter.")

    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler)
    logger = CSVLogger(save_dir=".", name=hyperparams["experiment_name"])

    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        logger=logger,
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=50,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm)


def finetune(hyperparams) -> None:
    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler)

    logger = CSVLogger(save_dir=".", name=hyperparams["experiment_name"])
    checkpoint_callback = ModelCheckpoint(
        monitor="val/rmse",
        mode="min",
        save_top_k=1,
    )

    stopper = EarlyStopping(
        monitor="val/rmse",
        mode="min",
        patience=100,
    )

    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        logger=logger,
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, stopper],
    )

    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm)


def test(hyperparams) -> None:
    if hyperparams["ckpt"] is None:
        raise ValueError("Testing expects a checkpoint, specify it using --ckpt=PATH.")

    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler, testing=True)
    logger = CSVLogger(save_dir=".", name=hyperparams["experiment_name"])
    trainer = pl.Trainer(
        logger=logger,
        deterministic=True,
    )
    trainer.logger.log_hyperparams(hyperparams)
    trainer.test(model, datamodule=dm)
