"""This module contains functions to create the plots for the report.
It includes a function to create a histogram of the molecular weight distribution of the datasets
and a function to create a UMAP projection of the MACCS keys fingerprints of the molecules in the datasets.

"""

import pandas as pd
from rdkit import Chem
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
import umap


def get_weight(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else None


def make_size_dist() -> None:
    lipo = pd.read_csv("data/Lipophilicity.csv")
    mfpcba = pd.read_csv("data/SD.csv")
    qm7 = pd.read_csv("data/QM7.csv")
    mfpcba["weight"] = mfpcba["smiles"].apply(get_weight)
    lipo["weight"] = lipo["smiles"].apply(get_weight)
    qm7["weight"] = qm7["smiles"].apply(get_weight)
    mfpcba["Dataset"] = "MF-PCBA"
    lipo["Dataset"] = "Lipophilicity"
    qm7["Dataset"] = "QM7"
    combined = pd.concat([lipo, mfpcba, qm7])

    fig, ax = plt.subplots()
    sns.histplot(
        combined,
        x="weight",
        fill=True,
        hue="Dataset",
        bins=50,
        common_norm=False,
        stat="percent",
        multiple="layer",
        legend=True,
        palette={"MF-PCBA": "lightgreen", "Lipophilicity": "blue", "QM7": "red"},
        hue_order=["MF-PCBA", "Lipophilicity", "QM7"],
        alpha=0.4,
    )
    ax.set_xbound(0, 800)
    ax.set_xlabel("Molecular Weight")
    ax.set_title("Molecular Weight Distribution of datasets")
    fig.savefig("results/figures/weight_distribution.png", dpi=300)


def make_maccs_keys(smiles: str) -> np.ndarray:
    fp = np.zeros((0,), dtype=int)
    ConvertToNumpyArray(GetMACCSKeysFingerprint(Chem.MolFromSmiles(smiles)), fp)
    return fp


def make_umap() -> None:
    np.random.seed(1337)

    lipo = pd.read_csv("data/Lipophilicity.csv")
    mfpcba = pd.read_csv("data/SD.csv")
    qm7 = pd.read_csv("data/QM7.csv")

    idx = np.arange(len(mfpcba.smiles))
    np.random.shuffle(idx)
    idx = idx[: len(idx) // 5]

    mfpcba_maccs = np.stack(mfpcba.smiles.iloc[idx].apply(make_maccs_keys).values)
    lipo_maccs = np.stack(lipo.smiles.apply(make_maccs_keys).values)
    qm7_maccs = np.stack(qm7.smiles.apply(make_maccs_keys).values)
    maccs = np.concatenate([mfpcba_maccs, lipo_maccs, qm7_maccs])
    mapper = umap.UMAP(n_components=2)
    labels = np.concatenate(
        [
            np.full(len(mfpcba_maccs), "MF-PCBA"),
            np.full(len(lipo_maccs), "Lipophilicity"),
            np.full(len(qm7_maccs), "QM7"),
        ]
    )
    mapper = mapper.fit(maccs)
    embeddings = mapper.transform(maccs)
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=labels,
        palette={"MF-PCBA": "lightgreen", "Lipophilicity": "blue", "QM7": "red"},
        s=2,
        ax=ax,
    )

    ax.set_title("UMAP projection of MACCS keys fingerprints")
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    leg = ax.legend()

    for handle in leg.legend_handles:
        handle.set_markersize(10)
    fig.savefig("results/figures/umap_projection.png", dpi=300)


make_size_dist()
make_umap()
