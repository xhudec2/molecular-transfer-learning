"""This script processes the results of the experiments and generates summary tables for the report.

It performs the following tasks:
- Loading the results from the specified directory and compiles them into a single DataFrame.
- Computes the mean and standard deviation of the metrics (MAE, RMSE, R2) for each model and dataset.
- Performs t-tests to compare the performance of different models
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_results(root: Path) -> pd.DataFrame:
    datasets = ["dr", "lipo", "qm7"]

    df_samples = []
    for dataset in datasets:
        dataset_dir = root / dataset
        for subdir in dataset_dir.iterdir():
            model = subdir.name
            maes, rmses, r2s = [], [], []
            for file in subdir.iterdir():
                if file.stem == "metrics":
                    df = pd.read_csv(file)
                    maes = df["test/mae"]
                    rmses = df["test/rmse"]
                    r2s = df["test/r2"]
                else:
                    df = pd.read_csv(file / "metrics.csv")
                    maes.append(df["test/mae"][0])
                    rmses.append(df["test/rmse"][0])
                    r2s.append(df["test/r2"][0])
            df_samples.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "maes": np.array(maes),
                    "rmses": np.array(rmses),
                    "r2s": np.array(r2s),
                }
            )

    df = pd.DataFrame(df_samples)
    df["maes_mean"] = df["maes"].apply(lambda x: np.mean(x))
    df["rmses_mean"] = df["rmses"].apply(lambda x: np.mean(x))
    df["r2s_mean"] = df["r2s"].apply(lambda x: np.mean(x))
    df["maes_std"] = df["maes"].apply(lambda x: np.std(x, ddof=1))
    df["rmses_std"] = df["rmses"].apply(lambda x: np.std(x, ddof=1))
    df["r2s_std"] = df["r2s"].apply(lambda x: np.std(x, ddof=1))
    path = root / "tables/model_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def perform_t_test(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    subset = df[df["dataset"] == dataset]
    models = subset["model"].unique()

    rows = []
    print("Performing t-tests for dataset:", dataset, "on RMSE")
    for i in range(len(models)):
        model1 = models[i]
        for j in range(i + 1, len(models)):
            model2 = models[j]
            scores1 = subset[subset["model"] == model1]["rmses"].values[0]
            scores2 = subset[subset["model"] == model2]["rmses"].values[0]
            scores1 = [x.strip() for x in scores1[1:-1].split(" ")]
            scores2 = [x.strip() for x in scores2[1:-1].split(" ")]
            scores1 = np.array(list(filter(lambda x: x != "", scores1))).astype(float)
            scores2 = np.array(list(filter(lambda x: x != "", scores2))).astype(float)
            t_stat, p_value = ttest_ind(scores1, scores2, equal_var=False)
            print(
                f"{model1:<14} vs {model2:<14}: t-statistic = {t_stat:.3f}, p-value = {p_value}"
            )
            rows.append(
                {
                    "model1": model1,
                    "model2": model2,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                }
            )
    print()
    return pd.DataFrame(rows)


def perform_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for dataset in df["dataset"].unique():
        rows = perform_t_test(df, dataset)
        all_rows.append(rows)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv("results/tables/t_test_results.csv", index=False)


def get_metrics(
    df_train: pd.DataFrame, df_test: pd.DataFrame, label: str
) -> tuple[float, float]:
    y = df_test[label]
    preds = np.ones_like(y) * np.mean(df_train[label])
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"RMSE: {rmse:10.3f}, MAE: {mae:10.3f}")
    return mae, rmse


def get_mean_metrics() -> None:
    df = []
    lipo_df_test = pd.read_csv("data/lipo_splits/lipo_test.csv")
    lipo_df_train = pd.read_csv("data/lipo_splits/lipo_train.csv")
    print("Lipo: ", end="")
    mae, rmse = get_metrics(lipo_df_train, lipo_df_test, "exp")
    df.append({"dataset": "lipo", "mae": mae, "rmse": rmse})

    qm7_df_test = pd.read_csv("data/qm7_splits/qm7_test.csv")
    qm7_df_train = pd.read_csv("data/qm7_splits/qm7_train.csv")
    print("QM7:  ", end="")
    mae, rmse = get_metrics(qm7_df_train, qm7_df_test, "u0_atom")
    df.append({"dataset": "qm7", "mae": mae, "rmse": rmse})

    dr_df_test = pd.read_csv("data/dr_splits/dr_test.csv")
    dr_df_train = pd.read_csv("data/dr_splits/dr_train.csv")
    print("DR:   ", end="")
    mae, rmse = get_metrics(dr_df_train, dr_df_test, "DR")
    df.append({"dataset": "dr", "mae": mae, "rmse": rmse})
    df = pd.DataFrame(df)
    df.to_csv("results/tables/mean_metrics.csv", index=False)


get_mean_metrics()
load_results(Path("results"))
perform_t_tests(pd.read_csv("results/tables/model_summary.csv"))
