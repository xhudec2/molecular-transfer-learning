# vgae_model

This module contains the data pipeline, model definition, and training entrypoints for the cross-domain molecular transfer learning experiments in this repository.

## What gets written to disk during training

For each run, Lightning logs to `./lightning_logs/<experiment-name>/`:

- `hparams.yaml`: run configuration (dataset paths, lr, seed, etc.)
- `metrics.csv`: logged metrics
- `checkpoints/`: saved `.ckpt` files (when checkpointing is enabled)

## Modules

- CLI: `vgae_model/__main__.py`
- Training: `vgae_model/utils/trainer.py`
- Default hyperparameters and dataset paths: `vgae_model/utils/hyperparams.py`
- Data:
  - `vgae_model/data/dataset.py`: Molecular graph datasets handling
  - `vgae_model/data/datamodule.py`: Lightning DataModule to wrap all datasets
  - `vgae_model/data/transforms.py`: Atom featurization and SMILES preprocessing
- Model:
  - `vgae_model/modelling/vgaemodule.py`: LightningModule combining a VGAE backbone and regression head
  - `vgae_model/modelling/backbone.py`: VGAE + set transformer aggregation
  - `vgae_model/modelling/regressor.py`: FFN Regression head
  - `vgae_model/modelling/gcn.py`: GCN encoder for VGAE

## Reproducibility

- The code uses Lightning’s `deterministic=True` for training.
- Each run’s seed is saved to `hparams.yaml`.
- For the exact same results, rerun with the same seed and the same environment installed using `uv`.
