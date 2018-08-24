"""
Functions that help load data from numpy files
"""
from pathlib import Path
import numpy as np


def load(dataset_name, algo_names, split_ids, predictions_path="../predictions",
         data_path="../data_files"):
    """Load data for a dataset and several algorithms"""
    # first, load dataset
    datasets = load_dataset(dataset_name, split_ids, data_path)
    # then load predictions
    predictions = load_predictions(algo_names, dataset_name, split_ids, predictions_path)
    return datasets, predictions


def load_dataset(dataset_name, split_ids, data_path):
    """Load data for a dataset"""
    datasets = {}
    for split_id in split_ids:
        dataset_path = Path(data_path) / Path(f"{dataset_name}_{split_id}.npz")
        datasets[split_id] = dict_from_npz(dataset_path)
    return datasets


def load_predictions(algo_names, dataset_name, split_ids, predictions_path):
    """Load predictions for the given algorithms"""
    predictions = {}
    for algo_name in algo_names:
        predictions[algo_name] = {}
        for split_id in split_ids:
            pred_path = Path(predictions_path) / Path(f"{dataset_name}_{split_id}_{algo_name}.npz")
            predictions[algo_name][split_id] = dict_from_npz(pred_path)
    return predictions


def dict_from_npz(npz_path):
    with np.load(npz_path) as npz_data:
        data = {k: npz_data[k] for k in npz_data.files}
    return data
