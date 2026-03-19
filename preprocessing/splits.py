import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
from rdkit import RDLogger
from pathlib import Path

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def get_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        if scaffold_smiles == "" or scaffold.GetNumAtoms() == 0:
            return "ACYCLIC"
        return scaffold_smiles
    except Exception as e:
        print(f"Failed with {e}")
        return "ACYCLIC"


def scaffold_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_state)

    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)

    n_total = len(df)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(indices) > n_test * 1.1:
            train_idx.extend(indices)
        elif len(test_idx) < n_test:
            test_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def make_dr(path: Path) -> None:
    df = pd.read_csv(path)
    df = df.dropna()
    df.to_csv(str(path).replace("sd", "dr"), index=False)


def main() -> None:
    root = Path("data")
    for path in root.glob("*.csv"):
        print(f"Processing {path.stem}")
        prefix = path.stem.lower()[:4]
        df = pd.read_csv(path)
        df["scaffold"] = df["smiles"].apply(get_scaffold)
        train_idx, val_idx, test_idx = scaffold_split(df)

        train_scaffolds = set(df.iloc[train_idx]["scaffold"])
        val_scaffolds = set(df.iloc[val_idx]["scaffold"])
        test_scaffolds = set(df.iloc[test_idx]["scaffold"])
        overlap_tv = len(train_scaffolds & val_scaffolds)
        overlap_tt = len(train_scaffolds & test_scaffolds)
        overlap_vt = len(val_scaffolds & test_scaffolds)
        assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0

        split_dir = root / f"{prefix}_splits"
        split_dir.mkdir(exist_ok=True)
        df.iloc[train_idx].to_csv(split_dir / f"{prefix}_train.csv", index=False)
        df.iloc[val_idx].to_csv(split_dir / f"{prefix}_val.csv", index=False)
        df.iloc[test_idx].to_csv(split_dir / f"{prefix}_test.csv", index=False)

        if prefix == "sd":
            (root / "dr_splits").mkdir(exist_ok=True)
            make_dr(split_dir / f"{prefix}_train.csv")
            make_dr(split_dir / f"{prefix}_val.csv")
            make_dr(split_dir / f"{prefix}_test.csv")


if __name__ == "__main__":
    main()
