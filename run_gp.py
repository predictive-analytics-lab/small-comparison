#!/usr/bin/env python
"""
Run GP models
"""
from pathlib import Path
from ugp_algorithm import UGP, UGPDemPar, UGPEqOpp

RESULTS_DIR = "./results/"
ALGOS = [
    UGP(s_as_input=True),
    UGP(s_as_input=False),
]

DATA_FILES = [
    "./adult_sex_1.npz",
]

def main():
    results_dir = Path(RESULTS_DIR)

    for data_file in DATA_FILES:
        data_path = Path(data_file)
        for algo in ALGOS:
            name = algo.get_name()
            algo.run(data_path, results_dir / Path(f"{name}_{data_path.stem}"))


if __name__ == "__main__":
    main()
