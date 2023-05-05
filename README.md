# dtree2dset

Converting Decision Trees into Decision Sets.

## Installation

1. Make sure you have [Python 3.9](https://www.python.org/downloads/) installed.
2. Run `pip install -r requirements.txt` to install the project dependencies.

## Usage

Run `python3 run_dt2ds.py -bench pmlb.txt` to reproduce the experiments.

## Description

1. folder `datasets` contains all datasets used in the paper.
2. folder `dt_models` contains all pre-trained decision tree models.
3. folder `ds_files` contains all output decision set models in .txt format.
4. `dataset_split.py` is used to split the datasets into training and testing sets.
5. `decision_tree.py` contains the implementation of decision tree classifiers.
6. `train_dts.py` is used to train decision tree models from datasets.
7. `tree2rules.py` implements the algorithm for converting decision trees into decision sets.
8. file `train_info.txt` collects the train/test accuracy of each decision tree model.
9. file `dt2ds_results.txt` collects all the results reported in the paper.
