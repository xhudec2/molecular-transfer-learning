"""Lightning DataModule for molecular graphs.

Creates `GraphMoleculeDataset` instances for train/val/test CSVs and returns
PyTorch Geometric DataLoaders. If `use_standard_scaler=True`, fits a
`StandardScaler` on the training labels and uses it to normalize targets.

The code is based on https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/data_loading.py
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import lightning as pl

from torch_geometric.loader import DataLoader as GeometricDataLoader  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from typing import Union, List, Tuple, Optional

from vgae_model.data.dataset import GraphMoleculeDataset


class GeometricDataModule(pl.LightningDataModule):
    """DataModule for graph molecules."""

    def __init__(
        self,
        batch_size: int,
        seed: int,
        max_atom_num: int = 80,
        split: Tuple[float, float] = (0.9, 0.05),
        train_path: Optional[str] = None,
        separate_valid_path: Optional[str] = None,
        separate_test_path: Optional[str] = None,
        id_column: Optional[str] = None,
        num_cores: Tuple[int, int, int] = (12, 0, 12),
        label_column_name: Union[str, List[str]] = "SD",
        lbl_or_emb: str = "lbl",
        smiles_column: str = "smiles",
        use_standard_scaler: bool = False,
    ) -> None:
        super().__init__()
        assert lbl_or_emb in [None, "lbl", "emb"]
        self.dataset: None | GraphMoleculeDataset = None
        self.train_path = train_path
        self.batch_size = batch_size
        self.seed = seed
        self.max_atom_num = max_atom_num
        self.split = split
        self.num_cores = num_cores
        self.separate_valid_path = separate_valid_path
        self.separate_test_path = separate_test_path
        self.label_column_name = label_column_name
        self.lbl_or_emb = lbl_or_emb
        self.id_column = id_column
        self.smiles_column = smiles_column

        self.use_standard_scaler = use_standard_scaler

        self.scaler = None
        if self.use_standard_scaler:
            train_df = pd.read_csv(self.train_path)
            train_data = train_df[self.label_column_name].values

            scaler = StandardScaler()
            if train_data.ndim == 1:
                scaler = scaler.fit(np.expand_dims(train_data, axis=1))
            else:
                scaler = scaler.fit(train_data)

            del train_data
            del train_df

            self.scaler = scaler

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def prepare_data(self) -> None:
        self.val = None
        self.test = None
        if self.train_path:
            self.dataset = GraphMoleculeDataset(
                csv_path=self.train_path,
                max_atom_num=self.max_atom_num,
                label_column_name=self.label_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
                smiles_column=self.smiles_column,
            )

            self.num_atom_features = self.dataset.num_atom_features

        if self.separate_valid_path:
            self.val = GraphMoleculeDataset(
                csv_path=self.separate_valid_path,
                max_atom_num=self.max_atom_num,
                label_column_name=self.label_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
                smiles_column=self.smiles_column,
            )

        if self.separate_test_path:
            self.test = GraphMoleculeDataset(
                csv_path=self.separate_test_path,
                max_atom_num=self.max_atom_num,
                label_column_name=self.label_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
                smiles_column=self.smiles_column,
            )

    def setup(self, stage: None | str = None) -> None:
        # Called on every GPU
        # Assumes prepare_data has been called
        self.train = self.dataset

    def train_dataloader(self, shuffle: bool = True) -> GeometricDataLoader | None:
        if self.train:
            return GeometricDataLoader(
                self.train,
                self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_cores[0],
                pin_memory=True,
                drop_last=True,
            )
        return None

    def val_dataloader(self) -> GeometricDataLoader:
        return GeometricDataLoader(
            self.val,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0 if not self.num_cores else self.num_cores[1],
        )

    def test_dataloader(self) -> GeometricDataLoader | None:
        if self.test:
            return GeometricDataLoader(
                self.test,
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0 if not self.num_cores else self.num_cores[2],
            )
        return None
