"""
Functions for computing metrics
"""
import numpy as np


def bin_data(data, num_bins):
    """Put the given data into bins

    Args:
        data: the data to put into bins
        num_bins: the number of bins
    Returns:
        an array of the same size as data with the corresponding bin indices and an array of values
        that are associated with each bin
    """
    # separate the data range into equal sized bins
    # nonlinear methods would also be possible
    bin_edges = np.linspace(data.min(), data.max(), num_bins + 1)
    bin_indices = np.digitize(data, bin_edges, right=True)
    # we associate a value with each bin by taking the average of the bin edges
    bin_values = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_indices, bin_values


def binned_positive_label_prob(labels, bin_indices, num_bins):
    """Compute the probability of a positive label for binned data

    Args:
        labels: the labels with which to compute the probability
        bin_indices: the binned data
        num_bins: the number of bins in the binned data
    Returns:
        an array with the positive label probability for each bin
    """
    accuracy = np.zeros(num_bins)
    for i in range(num_bins):
        bin_index = i + 1
        accuracy[i] = np.mean(labels[bin_indices == bin_index])
    return accuracy


def apply_to_all(func, results, datasets):
    """Apply the given function to all results

    Args:
        func: the function to apply
        results: nested dictionary where the nested levels are: algorithm name, sensitive attribute
                 and split ID
        datasets: nested dictionary where the nested levels are: sensitive attribute and split ID
    Returns:
        a nested dictionary with the same structure as `results` that contains the output of the
        given function
    """
    output = {}
    for algo in results:
        output[algo] = {}
        for sensitive in results[algo]:
            output[algo][sensitive] = {}
            for split_id in results[algo][sensitive]:
                output[algo][sensitive][split_id] = func(
                    results[algo][sensitive][split_id], datasets[sensitive][split_id])
    return output


def apply_to_merged_splits(func, results, datasets):
    """Apply the given function to results and datasets but with merged split IDs

    Args:
        func: the function to apply
        results: nested dictionary where the nested levels are: algorithm name, sensitive attribute
                 and split ID
        datasets: nested dictionary where the nested levels are: sensitive attribute and split ID
    Returns:
        a nested dictionary with the same structure as `results` except without the split ID level
        that contains the output of the given function
    """
    output = {}
    for algo in results:
        output[algo] = {}
        for sensitive in results[algo]:
            results_list = []
            datasets_list = []
            for split_id in results[algo][sensitive]:
                results_list.append(results[algo][sensitive][split_id])
                datasets_list.append(datasets[sensitive][split_id])
            merged_results = merge_all_entries(results_list)
            merged_datasets = merge_all_entries(datasets_list)
            output[algo][sensitive] = func(merged_results, merged_datasets)
    return output


def merge_all_entries(list_of_dictionaries):
    """Given a list of dictionaries with the same entries, a single dictionary is created"""
    out = {}
    for key in list_of_dictionaries[0]:
        list_of_entries = [dictionary[key] for dictionary in list_of_dictionaries]
        out[key] = np.concatenate(list_of_entries, axis=0)
    return out


def meta_confidence_and_accuracy(num_bins, s=None):
    """Returns a function that compute the confidence and the accuracy

    Args:
        num_bins: number of bins that are used to split the data based on the predicted score
        s: (optional) sensitive attribute to use as a filter
    Returns:
        function that computes the confidence and accuracy
    """
    def _confidence_and_accuracy(result, dataset):
        scores = result['pred_mean']
        labels = dataset['ytest']
        sensitive = dataset['stest']
        if s is not None:
            indices, confidence = bin_data(scores[sensitive == s], num_bins)
            accuracy = binned_positive_label_prob(labels[sensitive == s], indices, num_bins)
        else:
            indices, confidence = bin_data(scores, num_bins)
            accuracy = binned_positive_label_prob(labels, indices, num_bins)
        return confidence, accuracy
    return _confidence_and_accuracy


def tpr_diff_and_accuracy(result, dataset):
    """Compute TPR difference and accuracy"""
    # gather data
    scores = result['pred_mean']
    labels = dataset['ytest']
    sensitive = dataset['stest']
    # P(yhat=1|y=1, s=0)
    tpr_s0 = np.mean(scores[(labels == 1) & (sensitive == 0)] > 0.5)
    # P(yhat=1|y=1, s=1)
    tpr_s1 = np.mean(scores[(labels == 1) & (sensitive == 1)] > 0.5)
    accuracy = np.mean((scores > 0.5).astype(np.int) == labels)
    return abs(tpr_s0 - tpr_s1), accuracy
