import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import warnings
from rdkit import RDLogger
from pathlib import Path

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


RANDOM_SEED = 42
MAX_ATOM_NUM = 53
MAX_ATOMS_IN_MOL = 124
np.random.seed(RANDOM_SEED)


def get_max_atomic_number(smiles: str) -> int:
    max_num = max(
        Chem.MolFromSmiles(smiles).GetAtoms(), key=lambda x: x.GetAtomicNum()
    ).GetAtomicNum()
    return max_num


def get_num_atoms(smiles: str) -> int:
    return len(Chem.MolFromSmiles(smiles).GetAtoms())


def standardize_smiles(smiles: str) -> None | str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Failed with {e}")
        return None


def standardize_data(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    original_len = len(df)
    df["smiles"] = df["smiles"].apply(standardize_smiles)
    df = df[["smiles"] + feature_cols]
    df = df[~df["smiles"].isna()]
    df = df[df["smiles"].apply(get_max_atomic_number) <= MAX_ATOM_NUM]
    df = df[df["smiles"].apply(get_num_atoms) <= MAX_ATOMS_IN_MOL]

    failed = original_len - len(df)
    print("\nStandardization results:")
    print(f"  Original: {original_len} molecules")
    print(f"  Standardized: {len(df)} molecules")
    print(f"  Failed: {failed} molecules")
    return df


def deduplicate_data(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    n_duplicates = df["smiles"].duplicated().sum()
    print(f"Duplicate SMILES after standardization: {n_duplicates}")

    if n_duplicates > 0:
        dup_smiles = df[df["smiles"].duplicated(keep=False)]["smiles"].unique()
        print(f"\nFound {len(dup_smiles)} unique SMILES that appear multiple times:")

        for smi in dup_smiles:
            dups = df[df["smiles"] == smi]
            print(f"\n  SMILES: {smi}")
            for feature in features:
                print(f"  {feature} values: {dups[feature].tolist()}")

    aggs = {feature: "mean" for feature in features}
    deduped_df = df.groupby("smiles").agg(aggs).reset_index()
    print(f"After deduplication: {len(deduped_df)} molecules")
    return deduped_df


def remove_overlaps(datasets_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = list(datasets_dir.glob("*.csv"))
    for i in range(len(dfs) - 1):
        df1 = pd.read_csv(dfs[i])
        overlaps_all = np.zeros(len(df1), dtype=np.bool)
        for j in range(i + 1, len(dfs)):
            print(f"Removing overlaps from {dfs[i].stem} and {dfs[j].stem}.")
            df2 = pd.read_csv(dfs[j])
            overlaps = df1.smiles.isin(df2.smiles)
            overlaps_all |= overlaps
            print(f"Found and removed {overlaps.sum()} overlapping smiles\n")
        df1[~overlaps_all].to_csv(dfs[i])


def main() -> None:
    root = Path("data")
    features = {"SD": ["SD", "DR"], "Lipophilicity": ["exp"], "qm7": ["u0_atom"]}
    for dataset in root.glob("*.csv"):
        df = pd.read_csv(dataset)
        if dataset.stem == "SD":
            df = df.rename(columns={"neut-smiles": "smiles"})
        df = standardize_data(df, features[dataset.stem])
        df = deduplicate_data(df, features[dataset.stem])
        df.to_csv(dataset)
    remove_overlaps(root)


if __name__ == "__main__":
    main()
