import os
import joblib
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

from CToxPred.pairwise_correlation import CorrelationThreshold
from CToxPred.utils import compute_fingerprint_features, \
    compute_descriptor_features, compute_metrics
from CToxPred.hERG_model import hERGClassifier
from CToxPred.nav15_model import Nav15Classifier
from CToxPred.cav12_model import Cav12Classifier



def _generate_predictions(smiles_list: List[str]) -> None:
    """
    Generates predictions for hERG, Nav1.5, and Cav1.2 targets based on the provided list of SMILES.

    This function processes the input SMILES list and computes fingerprint and descriptor features for each compound.
    Then, it loads pre-trained models for hERG, Nav1.5, and Cav1.2 targets, and predicts the activity of each compound
    for these targets using the respective models. The predictions are saved to a CSV file named 'predictions.csv'
    with columns: 'SMILES', 'hERG', 'Nav1.5', and 'Cav1.2'. The 'hERG', 'Nav1.5', and 'Cav1.2' columns contain the
    binary predictions (0 or 1) for each target, representing non-toxic (negative class) or toxic (positive class)
    compounds, respectively.

    Parameters:
        smiles_list: List[str] 
            A list containing SMILES strings of chemical compounds.

    Returns:
        None: 
            The function saves the predictions to a CSV file named 'predictions.csv'.
    """
    # Compute features
    print('>>>>>>> Calculate Features <<<<<<<')
    fingerprints = compute_fingerprint_features(smiles_list)
    descriptors = compute_descriptor_features(smiles_list)
    # Process hERG
    print('>>>>>>> Predict hERG <<<<<<<')
    hERG_fingerprints = fingerprints
    ## Load model
    hERG_predictor = hERGClassifier(1905, 2)
    path = ['CToxPred', 'models', 'model_weights', 'hERG',
            '_herg_checkpoint.model']
    hERG_predictor.load(os.path.join(*path))
    device = torch.device('cpu')
    hERG_predictions = hERG_predictor(
        torch.from_numpy(hERG_fingerprints).float().to(device)).argmax(1).cpu()

    # Process Nav1.5
    print('>>>>>>> Predict Nav1.5 <<<<<<<')
    nav15_fingerprints = fingerprints
    nav15_descriptors = descriptors
    ## Load preprocessing pipeline
    path = ['CToxPred', 'models', 'decriptors_preprocessing', 'Nav1.5',
            'nav_descriptors_preprocessing_pipeline.sav']
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    nav15_descriptors = descriptors_transformation_pipeline.transform(
        nav15_descriptors)
    nav15_features = np.concatenate((nav15_fingerprints, nav15_descriptors),
                                    axis=1)
    ## Load model
    nav15_predictor = Nav15Classifier(2454, 2)
    path = ['CToxPred', 'models', 'model_weights', 'Nav1.5',
            '_nav15_checkpoint.model']
    nav15_predictor.load(os.path.join(*path))
    nav15_predictions = nav15_predictor(
        torch.from_numpy(nav15_features).float().to(device)).argmax(1).cpu()

    # Process Cav1.2
    print('>>>>>>> Predict Cav1.2 <<<<<<<')
    cav12_fingerprints = fingerprints
    cav12_descriptors = descriptors
    ## Load preprocessing pipeline
    path = ['CToxPred', 'models', 'decriptors_preprocessing', 'Cav1.2',
            'cav_descriptors_preprocessing_pipeline.sav']
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    cav12_descriptors = descriptors_transformation_pipeline.transform(
        cav12_descriptors)
    cav12_features = np.concatenate((cav12_fingerprints, cav12_descriptors),
                                    axis=1)
    ## Load model
    cav12_predictor = Cav12Classifier(2586, 2)
    path = ['CToxPred', 'models', 'model_weights', 'Cav1.2',
            '_cav12_checkpoint.model']
    cav12_predictor.load(os.path.join(*path))
    cav12_predictions = cav12_predictor(
        torch.from_numpy(cav12_features).float().to(device)).argmax(1).cpu()

    # Generate output
    results = pd.DataFrame({'SMILES': smiles_list, 'hERG': hERG_predictions,
                            'Nav1.5': nav15_predictions,
                            'Cav1.2': cav12_predictions})

    results.to_csv('predictions.csv', index=False)

def _help():
    """
    Display the usage instructions for the ctoxpred.py script.

    Usage:
        python ctoxpred.py <smiles_input_file.smi>

    Note:
        This function prints the command format required to run the ctoxpred.py 
        script and provides information about the expected input file 
        format. Ensure to replace '<smiles_input_file.smi>' with the actual path
        to your SMILES input file.
    """

    print("\n CToxPred: A comprehensive cardiotoxicity prediction tool of small molecules \
            \n\t  on three targets: hERG, Nav1.5, Cav1.2 \n \
            \n\tTo get predictions, run the command as follows:\n\t    $ python ctoxpred.py <input_file>.smi \
            \n\nWhere <input_file> is the SMILES input file to the software, and has the extension .smi \n")

if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        _help()
    elif len(sys.argv) > 2:
        _help()
    else:
        if not sys.argv[1].endswith('.smi'):
            print('File extension wrong.')
        else:
            with open(sys.argv[1], 'r') as file:
                smiles_list = []
                for smiles in file:
                    smiles_list.append(smiles.strip())
            _generate_predictions(smiles_list)


