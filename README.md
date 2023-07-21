# CToxPred
Comprehensive cardiotoxicity prediction tool of small molecules on three targets: hERG, Nav1.5, Cav1.2

<p align="center">
	<img src="images/All-ConfusionMatix.png" />
</p>

The software cardiotoxicity predictions can be interpreted as follows (Potencies are in μM):
 - Blocker     (1) : 0 < IC50 <= 10
 - Non-blocker (0) : IC50 > 10


:exclamation:Clone first the whole repository package and follow the steps bellow.

## Prerequisites
1- Create and activate a conda environment:

		$conda create -n ctoxpred python=3.7
		$conda activate ctoxpred

2- Install packages:

		$bash install.sh

3- Clone the repository: 

		$git clone git@github.com:issararab/CToxPred.git

4- Move to the repository:

		$cd CToxPred

5- Run test:

		$python CToxPred.py data/test_smiles_list.smi
  


