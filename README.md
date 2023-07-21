# CToxPred
Comprehensive cardiotoxicity prediction tool of small molecules on three targets: hERG, Nav1.5, Cav1.2

<p align="center">
	<img src="images/All-ConfusionMatix.png" />
</p>

The model predicts Cardiotoxicities based on 2 classes (Potencies are in μM):
 - Blocker    : 0 < IC50 <= 01
 - Non-blocker: IC50 > 10


:exclamation:Clone first the whole repository package and follow the steps bellow.

## Prerequisites
2- Create and activate a conda environment:

		$conda create -n ctoxpred python=3.7
		$conda activate ctoxpred

3- Install packages:

		$bash install.sh

1- Clone the repository: 

		$git clone git@github.com:issararab/CToxPred.git

2- Move to the repository:

		$cd CToxPred

2- Run test:

		$python CToxPred.py data/test_smiles_list.smi
  


