# Results Directory

This directory contains all experimental results from the molecular transfer learning project, including model training results, hyperparameter optimization trials, and evaluation metrics across multiple datasets.

## Structure

### Dataset Results

This includes directiories: `dr`, `lipo`, and `qm7`.
Results are organized by dataset:

- **dr** — Drug Response dataset results
- **lipo** — Lipophilicity dataset results  
- **qm7** — QM7 dataset results

Each dataset directory contains three subdirectories:

#### Model Subdirectories

- **non-pretrained** — Models trained from scratch without transfer learning
  - Contains numbered model directories (model_0/ to model_4/) for the trained models
  - Each model_X/ directory contains:
    - `hparams.yaml` — Hyperparameters used for training
    - `metrics.csv` — Evaluation metrics (MAE, RMSE, R2) on test set

- **pretrained** — Models initialized with pretrained backbone
  - Same structure as non-pretrained
  - Used to evaluate transfer learning benefits

- **rf** — Random Forest baseline
  - `metrics.csv` — Single set of metrics for the RF model

### Hyperparameter Optimization

This includes the `optuna` directory.
Hyperparameter tuning results:

```
optuna/
├── non-pretrained-model/
│   ├── dr_hpopt/
│   ├── lipo_hpopt/
│   └── qm7_hpopt/
└── pretrained-model/
    ├── dr_hpopt/
    ├── lipo_hpopt/
    └── qm7_hpopt/
```

Each directory contains the best learning rate and the optimization history plot.

### Tabular data

This includes the `tables` directory.

- **model_summary.csv** — Aggregated results across all models
  - Columns: dataset, model, maes, rmses, r2s (arrays), plus mean and std for each metric
  - Used for comparisons in the report
- **mean_metrics.csv** - Metrics for predicting mean on the given datasets
- **t_test_results.csv** - T test statistical model comparison results

### Preprocessing Logs

This includes the `preprocess_logs` directory.  
Logs from data preprocessing and splitting operations.

### Figures

This includes the `tables` directory.
Visualizations dataset used in the report.
