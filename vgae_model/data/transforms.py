# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/chemprop_featurisation.py
from rdkit import Chem
from typing import Sequence, Dict


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)


def onek_encoding_unk(value: int, choices: Sequence):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(
    atom: Chem.rdchem.Atom,
    features_constants: Dict[str, Sequence],
    functional_groups=None,
):
    features = (
        onek_encoding_unk(atom.GetAtomicNum(), features_constants["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), features_constants["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), features_constants["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), features_constants["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), features_constants["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), features_constants["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def get_atom_constants(max_atomic_num: int):
    return {
        "atomic_num": list(range(max_atomic_num)),
        "degree": [0, 1, 2, 3, 4, 5],
        "formal_charge": [-1, -2, 1, 2, 0],
        "chiral_tag": [0, 1, 2, 3],
        "num_Hs": [0, 1, 2, 3, 4],
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    }
