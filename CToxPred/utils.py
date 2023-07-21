from math import sqrt
from typing import Tuple, List

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint, \
    CalculateECFP2Fingerprint
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from sklearn.metrics import f1_score, confusion_matrix


def compute_fingerprint_features(smiles_list: List[str]) -> np.ndarray:
    """
    Compute ECFP2 & PubChem fingerprint features for a list 
    of SMILES strings

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    np.ndarray
        Returns a 2D numpy array, where each row corrsponds
        to the fingerprints of a SMILES strings in order.
    """
    molecular_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    # Initialize an array to store ECFP2 & PubChem fingerprint features
    features = np.zeros((len(smiles_list), 1024 + 881), dtype=np.int32)

    for i, mol in enumerate(molecular_mols):
        ECFP2_mol_fingerprint = CalculateECFP2Fingerprint(mol)
        pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)
        numerical_representation = np.concatenate(
            (ECFP2_mol_fingerprint[0], pubchem_mol_fingerprint))
        features[i] = numerical_representation

    return features


def compute_descriptor_features(smiles_list: List[str]) -> pd.DataFrame:
    """
    Compute 2D descriptor features for a list of SMILES strings

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    np.ndarray
        Returns a pandas dataframe, where each row corrsponds
        to the descriptors of a SMILES strings in order.
    """
    descriptor_calc_2D = Calculator(descriptors, ignore_3D=True)
    molecular_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    descriptors_2D = descriptor_calc_2D.pandas(molecular_mols)
    return descriptors_2D


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the graph representation of SMILES string

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    Returns a tuple of numpy arrays.
    """
    # Mol
    smiles_mol = Chem.MolFromSmiles(smiles)
    num_atoms = len(smiles_mol.GetAtoms())
    # Adjacency list
    smiles_adj = Chem.rdmolops.GetAdjacencyMatrix(smiles_mol)
    # Node-Features Matrix
    smiles_atoms_features = []
    for atom in smiles_mol.GetAtoms():
        smiles_atoms_features.append(_atom_feature(atom))
    features_matrix = np.asarray(smiles_atoms_features)
    # Edge list
    # Extract row and column indices where adj matrix elements are non-zero
    row_indices, col_indices = np.where(smiles_adj != 0)
    edge_list = np.stack((row_indices, col_indices))
    return features_matrix, edge_list


def smiles_batch_to_graph(smiles_list: List[str]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Compute the graph representations for a list of SMILES strings.
    The function pads zeros, to match the size of the molecular
    structure with the largest number of atoms

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    Returns a tuple of numpy arrays.
    """
    adjacency_matrices = []
    features_matrices = []
    target_size = 0
    max_num_atoms = max(
        [len(Chem.MolFromSmiles(smiles).GetAtoms()) for smiles in smiles_list])
    for smiles in smiles_list:
        # Mol
        smiles_mol = Chem.MolFromSmiles(smiles)
        # Adjacency list
        smiles_adj_temp = Chem.rdmolops.GetAdjacencyMatrix(smiles_mol)
        # Graph sepresentation
        smiles_atoms_features = np.zeros((max_num_atoms, 67))
        smiles_adj = np.zeros((max_num_atoms, max_num_atoms))
        # Node-Features-Matrix
        smiles_atoms_features_temp = []
        for atom in smiles_mol.GetAtoms():
            smiles_atoms_features_temp.append(
                _atom_feature(atom))  ### atom features
        mol_dim = min(len(smiles_atoms_features_temp), max_num_atoms)
        smiles_atoms_features[:mol_dim, ] = smiles_atoms_features_temp[
                                            :mol_dim]
        features_matrices.append(smiles_atoms_features)
        # Adj-preprocessing
        smiles_adj[:mol_dim, :mol_dim] = smiles_adj_temp[:mol_dim,
                                         :mol_dim] + np.eye(mol_dim)
        # Extract row and column indices where matrix elements are non-zero
        row_indices, col_indices = np.where(smiles_adj != 0)
        adjacency_matrices.append(np.stack((row_indices, col_indices)))
        target_size = max(target_size, adjacency_matrices[-1].shape[1])
    # Convert to numpy arrays
    features_matrices = np.asarray(features_matrices)
    padded_adjacency_matrices = np.zeros((len(smiles_list), 2, target_size))
    for i, array in enumerate(adjacency_matrices):
        padded_adjacency_matrices[i, :, :target_size] = np.pad(array, (
        (0, 0), (0, target_size - array.shape[1])), mode='constant')
    return features_matrices, padded_adjacency_matrices


def _atom_feature(atom: Atom) -> np.ndarray:
    """
    Generate a one-hot encoding array representing various features 
    of an RDKit Atom object.

    Parameters:
        atom :Atom
            The RDKit Atom object for which the features will be encoded.

    Returns:
        np.ndarray: 
        A one-hot encoding array containing features of the input Atom.
            - The first part of the array encodes the atomic symbol using one-hot encoding.
            - The second part encodes the degree (number of neighbors) of the Atom.
            - The third part encodes the total number of hydrogen (H) atoms bonded to the Atom.
            - The fourth part encodes the implicit valence of the Atom.
            - The fifth element encodes whether the Atom is aromatic (1 if aromatic, 0 otherwise).
            - The last part encodes ring information for the Atom.
    """
    return np.array(_one_of_k_encoding_unk(atom.GetSymbol(),
                                           ['C', 'N', 'O', 'F', 'Cl', 'S',
                                            'Br', 'P', 'I', 'Na', 'H', 'Si', 
                                            'Sn', 'Hg', 'B', 'Ca', 'Se', 'K', 
                                            'Zn', 'Bi', 'Cd', 'Cr', 'Li', 'In',
                                            'Sb', 'As', 'Fe', 'Cu', 'Pb', 'Mg', 
                                            'Al', 'V', 'Tl', 'Ag', 'Pd', 'Co',
                                            'Ti', 'Ge', 'Au','Ni', 'Mn', 'Pt']) +
                    _one_of_k_encoding(atom.GetDegree(),
                                       [0, 1, 2, 3, 4, 5, 6]) +
                    _one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    _one_of_k_encoding(atom.GetImplicitValence(),
                                       [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] + _get_ring_info(atom))


def _one_of_k_encoding(x: str, allowable_set: List[str]) -> List[int]:
    """
    One-hot encodes a string with respect to the allowable_set.

    Parameters:
        x: str 
            The input string to be encoded.
        allowable_set: list[str] 
            The list of allowable strings.

    Returns:
        Retruns a one-hot encoding array where each element 
        is True if the corresponding string in allowable_set 
        matches x, otherwise False.

    Raises:
        Exception: If the input x is not in the allowable_set.
    """
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def _one_of_k_encoding_unk(x: str, allowable_set: List[str]) -> List[int]:
    """
    One-hot encodes a string with respect to the allowable_set.
    If string doesn't match  any element, map it to the last element.

    Parameters:
        x: str 
            The input string to be encoded.
        allowable_set: list[str] 
            The list of allowable strings.

    Returns:
        Retruns a one-hot encoding array where each element 
        is True if the corresponding string in allowable_set 
        matches x, otherwise False.
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def _get_ring_info(atom: Atom) -> List[int]:
    """
    Generate a one-hot encoding array based on ring 
    information of an RDKit Atom object.

    Parameters:
        atom :Atom 
            The RDKit Atom object for which the ring information will be encoded.

    Returns:
        List[int]: 
            A one-hot encoding array representing the ring information of the input Atom.
            The array contains 1's at positions [3, 4, 5, 6, 7, 8] if the Atom is part 
            of a ring of size 3 to 8, respectively, and 0's at all other positions.
    """
    print(type(atom))
    ring_info_feature = []
    for i in range(3, 9):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)
    print(ring_info_feature)
    return ring_info_feature


def compute_metrics(ground_truth: List[int], predicted: List[int]) -> None:
    """
    Computes and prints binary classification performance metrics.

    Parameters:
        ground_truth: List[int] 
            The list of true labels (ground truth).
        predicted: List[int] 
            The list of predicted labels.

    Returns:
        None: 
            This function does not return any value; it prints the computed metrics.

    Metrics Computed and Printed:
        - Confusion Matrix
        - True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)
        - Accuracy (AC)
        - F1-score (f1)
        - Sensitivity (SN)
        - Specificity (SP)
        - Correct Classification Rate (CCR)
        - Matthews Correlation Coefficient (MCC)
    """
    print('Binary classification performace metrics:')
    print(confusion_matrix(ground_truth, predicted))
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted).ravel()
    print("TP, FN, TN, FP")
    print("{:02d}, {:02d}, {:02d}, {:02d}".format(tp, fn, tn, fp))
    print("AC: {0:.3f}".format((tp + tn) / (tp + tn + fn + fp)))
    print("f1: {0:.3f}".format(f1_score(ground_truth, predicted)))
    print("SN: {0:.3f}".format((tp) / (tp + fn)))
    print("SP: {0:.3f}".format((tn) / (tn + fp)))
    print("CCR: {0:.3f}".format((((tp) / (tp + fn)) + ((tn) / (tn + fp))) / 2))
    print("MCC: {0:.3f}".format((tp * tn - fp * fn) / (
        sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp)))))