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


def binned_logistic_accuracy(labels, bin_indices, num_bins):
    """Compute the logistic accuracy for binned data

    Args:
        labels: the labels with which to compute the accuracy
        bin_indices: the binned data
        num_bins: the number of bins in the binned data
    Returns:
        an array with the accuracy for each bin
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
