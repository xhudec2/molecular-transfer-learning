# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/data_loading.py
import torch
import pandas as pd  # type: ignore[import-untyped]

from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


from vgae_model.data.transforms import (
    get_atom_constants,
    atom_features,
    remove_smiles_stereo,
)


class GraphMoleculeDataset(TorchDataset[GeometricData]):
    def __init__(
        self,
        csv_path: str,
        max_atom_num: int,
        label_column_name: str | list[str],
        lbl_or_emb: str = "lbl",
        smiles_column: str = "smiles",
        scaler: None | StandardScaler = None,
        id_column: None | str = None,
    ) -> None:
        super().__init__()
        assert lbl_or_emb in [None, "lbl", "emb"]
        self.df = pd.read_csv(csv_path)
        self.label_column_name = label_column_name
        self.atom_constants = get_atom_constants(max_atom_num)
        self.num_atom_features = (
            sum(len(choices) for choices in self.atom_constants.values()) + 2
        )
        self.lbl_or_emb = lbl_or_emb
        self.scaler = scaler
        self.id_column = id_column
        self.smiles_column = smiles_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: torch.Tensor | slice | list[int]
    ) -> GeometricData | list[GeometricData]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            slice_step = idx.step if idx.step else 1
            idx = list(range(idx.start, idx.stop, slice_step))
        if not isinstance(idx, list):
            idx = [idx]

        selected = self.df.iloc[idx]
        smiles_ = selected[self.smiles_column].values

        targets = selected[self.label_column_name].values

        if self.scaler is not None:
            labels = torch.Tensor(self.scaler.transform(targets.reshape(-1, 1)))
        else:
            labels = torch.Tensor(targets)

        smiles = [remove_smiles_stereo(s) for s in smiles_]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        atom_feat = [
            torch.Tensor(
                [atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]  # type: ignore
            )
            for mol in rdkit_mols
        ]

        edge_index = []

        for mol in rdkit_mols:
            ei = torch.nonzero(
                torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))
            ).T

            edge_index.append(ei)

        geometric_data_points = [
            GeometricData(
                x=atom_feat[i],
                edge_index=edge_index[i],
                y=labels[i],
            )
            for i in range(len(atom_feat))
        ]

        for i, data_point in enumerate(geometric_data_points):
            data_point.smiles = smiles[i]

        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points
