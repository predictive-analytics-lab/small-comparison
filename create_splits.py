"""
Create splits of the datasets and save them as numpy files
"""
import sys
from pathlib import Path
import numpy as np

from data.objects.list import DATASETS, get_dataset_names
from data.objects.processed_data import ProcessedData

NUM_SPLITS = 5


def main(save_path, datasets=get_dataset_names(), num_splits=NUM_SPLITS):
    print(f"Datasets: '{datasets}'")
    tag = 'numerical-binsensitive'
    for dataset in DATASETS:
        if dataset.get_dataset_name() not in datasets:
            continue

        print("\nEvaluating dataset:" + dataset.get_dataset_name())

        all_sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
        for sensitive in dataset.get_sensitive_attributes():
            print(f"Sensitive attribute:{sensitive}")
            processed_dataset = ProcessedData(dataset)
            train_test_splits = processed_dataset.create_train_test_splits(num_splits, sensitive)
            for split_id, split in enumerate(train_test_splits[tag]):
                train, test = split
                privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
                positive_val = dataset.get_positive_class_val(tag)
                class_attr = dataset.get_class_attribute()
                raw_data = _prepare_data(
                    train, test, class_attr, positive_val, all_sensitive_attributes, sensitive,
                    privileged_vals
                )
                filename = f"{dataset}_{sensitive}_{split_id}"
                data_path = Path(save_path) / Path(filename + ".npz")
                np.savez(data_path, **raw_data)


def _prepare_data(train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
                  single_sensitive, privileged_vals):
    # Separate data
    sensitive = [df[single_sensitive].values[:, np.newaxis] for df in [train_df, test_df]]
    label = [df[class_attr].values[:, np.newaxis] for df in [train_df, test_df]]
    nosensitive = [df.drop(columns=sensitive_attrs).drop(columns=class_attr).values
                   for df in [train_df, test_df]]

    # Check sensitive attributes
    assert list(np.unique(sensitive[0])) == [0, 1] or list(np.unique(sensitive[0])) == [0., 1.]

    # Check labels
    label = fix_labels(label, positive_class_val)
    return dict(xtrain=nosensitive[0], xtest=nosensitive[1], ytrain=label[0], ytest=label[1],
                strain=sensitive[0], stest=sensitive[1])


def fix_labels(labels, positive_class_val):
    """Make sure that labels are either 0 or 1

    Args"
        labels: the labels as a list of numpy arrays
        positive_class_val: the value that corresponds to a "positive" predictions

    Returns:
        the fixed labels
    """
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1] and positive_class_val == 1:
        return labels
    elif label_values == [1, 2] and positive_class_val == 1:
        return [2 - y for y in labels]
    raise ValueError("Labels have unknown structure")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
