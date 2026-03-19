"""Lightning training.

This module wraps Lightning `Trainer` construction and provides the task
functions used by the CLI by the `vgae_model` module.

Functions:
- `pretrain`: train a model from scratch (no checkpoint)
- `finetune`: optionally load a checkpoint and train for supervised regression
- `test`: evaluate a checkpoint on the configured test split
- `optimize_lr`: Optuna learning-rate sweep
"""

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
import lightning as pl
from vgae_model.modelling import VGAEModule, VGAERegressionHead
from vgae_model.data import GeometricDataModule
from typing import Any
from sklearn.preprocessing import StandardScaler  # type: ignore
import optuna
import matplotlib.pyplot as plt
import os


def _get_datamodule(hyperparams: dict[str, Any]) -> GeometricDataModule:
    """Create the Lightning DataModule from the hyperparameter dict."""
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


def _get_model(
    hyperparams: dict[str, Any], scaler: StandardScaler, testing: bool = False
) -> VGAEModule:
    """Build a model (fresh or loaded from checkpoint).

    When `testing=False` and a checkpoint is provided, the regression head is
    re-initialized and the backbone GNN parameters are frozen to allow for finetuning.
    """
    if hyperparams["ckpt"] is None:
        return VGAEModule(
            num_features=hyperparams["max_atom_num"] + 27,
            lr=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            scaler=scaler,
            monitor_loss=hyperparams["monitor_loss"],
        )

    model = VGAEModule.load_from_checkpoint(
        hyperparams["ckpt"],
        num_features=hyperparams["max_atom_num"] + 27,
        lr=hyperparams["lr"],
        batch_size=hyperparams["batch_size"],
        scaler=scaler,
        monitor_loss=hyperparams["monitor_loss"],
    )

    if not testing:
        model.regressor = VGAERegressionHead()
        for param in model.backbone.gnn_model.parameters():
            param.requires_grad = False

    return model


def optimize_lr(hyperparams: dict[str, Any], n_trials: int = 10) -> None:
    """Optimize learning rate with Optuna and save a trial-history plot."""
    dm = _get_datamodule(hyperparams)

    def objective(trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-6, 0.001, log=True)
        trial_hyperparams = hyperparams.copy()
        trial_hyperparams["lr"] = lr

        model = _get_model(trial_hyperparams, dm.scaler)

        stopper = EarlyStopping(
            monitor="val/rmse",
            mode="min",
            patience=25,  # smaller patience than during finetuning to speedup hyperparam optimization
        )

        trainer = pl.Trainer(
            max_epochs=trial_hyperparams["max_epochs"],
            logger=False,
            deterministic=True,
            enable_checkpointing=False,
            log_every_n_steps=10,
            num_sanity_val_steps=0,
            callbacks=[stopper],
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, datamodule=dm)
        best_val_rmse = trainer.callback_metrics.get("val/rmse")
        return best_val_rmse.item()

    study = optuna.create_study(direction="minimize", study_name="lr_optimization")
    study.optimize(objective, n_trials=n_trials)

    save_dir = f"lightning_logs/{hyperparams['experiment_name']}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/best_trial.txt", "w") as f:
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        f.write(f"Best Validation RMSE: {study.best_value}\n")
        f.write("Best Params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")

    # plot creation
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    axes.plot([t.value for t in study.trials], "b-o", markersize=4)
    axes.axhline(
        study.best_value,
        color="r",
        linestyle="--",
        label=f"Best: {study.best_value:.4f}",
    )
    axes.set_xlabel("Trial")
    axes.set_ylabel("Validation RMSE")
    axes.set_title("Optimization History")
    axes.legend()
    save_path = f"{save_dir}/hpopt.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def pretrain(hyperparams: dict[str, Any]) -> None:
    """Pretrain a model from scratch on the configured dataset."""
    if hyperparams["ckpt"] is not None:
        raise ValueError("Pretraining does not expect a checkpoint parameter.")

    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler)
    logger = CSVLogger(save_dir=".", version=hyperparams["experiment_name"])

    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        logger=logger,
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=50,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    assert trainer.logger is not None  # shutup mypy

    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm)


def finetune(hyperparams: dict[str, Any]) -> None:
    """Finetune a model (either from random initialization or a pretrained one)."""
    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler)

    logger = CSVLogger(save_dir=".", version=hyperparams["experiment_name"])

    # We monitor val/rmse, as it seemed as the most "stable" metric during training
    checkpoint_callback = ModelCheckpoint(
        monitor="val/rmse",
        mode="min",
        save_top_k=1,
    )

    stopper = EarlyStopping(
        monitor="val/rmse",
        mode="min",
        patience=100,  # kept the same as in https://www.nature.com/articles/s41467-024-45566-8
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

    assert trainer.logger is not None  # shutup mypy

    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, datamodule=dm)


def test(hyperparams: dict[str, Any]) -> None:
    """Evaluate a checkpoint on the configured test split."""
    if hyperparams["ckpt"] is None:
        raise ValueError("Testing expects a checkpoint, specify it using --ckpt=PATH.")

    dm = _get_datamodule(hyperparams)
    model = _get_model(hyperparams, dm.scaler, testing=True)
    logger = CSVLogger(save_dir=".", version=hyperparams["experiment_name"])
    trainer = pl.Trainer(
        logger=logger,
        deterministic=True,
    )

    assert trainer.logger is not None  # shutup mypy

    trainer.logger.log_hyperparams(hyperparams)
    trainer.test(model, datamodule=dm)
