#!/usr/bin/env python
"""
Run GP models
"""
from pathlib import Path
from ugp_algorithm import UGP, UGPDemPar, UGPEqOpp

OUTPUT_DIR = "./predictions/"
DATA_DIR = "./data_files/"

ALGOS = [
    UGP(s_as_input=True),
    UGP(s_as_input=False),
    UGPEqOpp(s_as_input=False),
    UGPEqOpp(s_as_input=True),
]

DATASETS = [
    # format: (dataset_name, [sensitive_attributes], [split_ids])
    # ("two-gaussians", ["sensitive-attr"], [0, 1, 2, 3, 4]),
    ("adult", ["race", "sex"], [0, 1, 2]),
    ("propublica-recidivism", ["race", "sex"], [0, 1, 2]),
]


def main():
    results_dir = Path(OUTPUT_DIR)
    data_base_path = Path(DATA_DIR)

    for dataset_name, sensitives, split_ids in DATASETS:
        for sensitive in sensitives:
            for split_id in split_ids:
                dataset_descriptor = f"{dataset_name}_{sensitive}_{split_id}"
                data_path = data_base_path / Path(f"{dataset_descriptor}.npz")
                for algo in ALGOS:
                    name = algo.get_name()
                    algo.run(data_path, results_dir / Path(f"{dataset_descriptor}_{name}.npz"))


if __name__ == "__main__":
    main()
