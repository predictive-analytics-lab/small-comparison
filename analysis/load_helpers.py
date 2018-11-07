"""
Functions that help load data from numpy files
"""
from pathlib import Path
import numpy as np


def load(dataset_name, sensitives, split_ids, algo_names, predictions_path="../predictions",
         data_path="../data_files"):
    """Load data for a dataset and several algorithms"""
    # first, load dataset
    datasets = load_dataset(dataset_name, sensitives, split_ids, data_path)
    # then load predictions
    predictions = load_predictions(algo_names, dataset_name, sensitives, split_ids,
                                   predictions_path)
    return datasets, predictions


def load_dataset(dataset_name, sensitives, split_ids, data_path):
    """Load data for a dataset"""
    datasets = {}
    for sensitive in sensitives:
        datasets[sensitive] = {}
        for split_id in split_ids:
            dataset_path = Path(data_path) / Path(f"{dataset_name}_{sensitive}_{split_id}.npz")
            datasets[sensitive][split_id] = dict_from_npz(dataset_path)
    return datasets


def load_predictions(algo_names, dataset_name, sensitives, split_ids, predictions_path):
    """Load predictions for the given algorithms"""
    predictions = {}
    for algo_name in algo_names:
        predictions[algo_name] = {}
        for sensitive in sensitives:
            predictions[algo_name][sensitive] = {}
            for split_id in split_ids:
                pred_path = Path(predictions_path) / (
                    Path(f"{dataset_name}_{sensitive}_{split_id}_{algo_name}.npz"))
                predictions[algo_name][sensitive][split_id] = dict_from_npz(pred_path)
    return predictions


def dict_from_npz(npz_path):
    """Dictionary from npz file"""
    with np.load(npz_path) as npz_data:
        data = {k: npz_data[k] for k in npz_data.files}
    return data


def load_matching_algos(filter_func, predictions_path, dataset_name=None, sensitives=None,
                        split_ids=None):
    """Load predictions where the name of the algorithm returns true when passed to `filter_func`"""
    predictions = {}
    for file_path in Path(predictions_path).iterdir():
        if not file_path.is_file():
            continue
        name_parts = file_path.stem.split('_')  # the stem is the filename without file extension
        if len(name_parts) < 4:
            continue

        # optionally test whether this is the right dataset
        this_dataset_name = name_parts[0]
        sensitive = name_parts[1]
        split_id = name_parts[2]
        if dataset_name is not None and this_dataset_name != dataset_name:
            continue
        if sensitives is not None and sensitive not in sensitives:
            continue
        if split_ids is not None and split_id not in split_ids:
            continue

        # join everything together again except for the first three parts (dataset_name, sensitive
        # and split_id)
        algo_name = "_".join(name_parts[3:])
        if not filter_func(algo_name):  # test whether it's the correct algo name
            continue
        predictions[algo_name][sensitive][split_id] = dict_from_npz(file_path)
    return predictions
