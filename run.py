#!/usr/bin/env python
"""
Run GP models
"""
from pathlib import Path
from ugp_algorithm import UGP, UGPDemPar, UGPEqOpp

OUTPUT_DIR = "./predictions/"
DATA_DIR = "./data_files/"

algos = []
for tnr in [0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
    for tpr in [0.6, 0.7, 0.8, 0.9, 1.0]:
        algos.append(UGPEqOpp(use_lr=True, s_as_input=True, tnr0=tnr, tnr1=tnr, tpr0=tpr, tpr1=tpr))
        algos.append(UGPEqOpp(use_lr=True, s_as_input=False, tnr0=tnr, tnr1=tnr, tpr0=tpr, tpr1=tpr))
ALGOS = algos
# ALGOS = [
#     # UGP(s_as_input=True, use_lr=True),
#     # UGP(s_as_input=False, use_lr=True),
#     # UGPDemPar(s_as_input=True, use_lr=True),
#     # UGPDemPar(s_as_input=False, use_lr=True),
#     UGPEqOpp(s_as_input=False, use_lr=True),
#     UGPEqOpp(s_as_input=True, use_lr=True),
#     # UGPDemPar(s_as_input=True),
#     # UGPDemPar(s_as_input=False),
# ]

DATASETS = [
    # format: (dataset_name, [sensitive_attributes], [split_ids])
    # ("two-gaussians", ["sensitive-attr"], [0, 1, 2, 3, 4]),
    # ("adult", ["race", "sex"], [0, 1, 2, 3, 4]),
    ("propublica-recidivism", ["race", "sex"], [0, 1, 2, 3, 4]),
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
