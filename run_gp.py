#!/usr/bin/env python
"""
Run GP models
"""
from pathlib import Path
from ugp_algorithm import UGP, UGPDemPar, UGPEqOpp

OUTPUT_DIR = "./predictions/"
ALGOS = [
    UGP(s_as_input=True),
    UGP(s_as_input=False),
]

DATA_FILES = [
    "./data_files/two-gaussians_sensitive-attr_0.npz",
    # "./data_files/two-gaussians_sensitive-attr_1.npz",
    # "./data_files/two-gaussians_sensitive-attr_2.npz",
    # "./data_files/two-gaussians_sensitive-attr_3.npz",
    # "./data_files/two-gaussians_sensitive-attr_4.npz",
]


def main():
    results_dir = Path(OUTPUT_DIR)

    for data_file in DATA_FILES:
        data_path = Path(data_file)
        for algo in ALGOS:
            name = algo.get_name()
            algo.run(data_path, results_dir / Path(f"{data_path.stem}_{name}.npz"))


if __name__ == "__main__":
    main()
