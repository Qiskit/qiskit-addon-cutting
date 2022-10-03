#!/usr/bin/env python

# NOTE: This script is meant to be run from the directory it is located in

# TODO: We might as well call papermill directly from python, instead of using
# the CLI via subprocess

import subprocess
from pathlib import Path


def main():
    output_dir = Path("output-notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_molecular_orbitals in (2, 4, 6, 8, 10):
        for structure in ("reactant", "product"):
            for use_parallel in (True, False):
                print((structure, n_molecular_orbitals, use_parallel))
                mode = "parallel" if use_parallel else "serial"
                subprocess.run(
                    [
                        "papermill",
                        "entanglement_forging_ckt_demo.ipynb",
                        output_dir
                        / f"{structure}_{n_molecular_orbitals}_{mode}_0.ipynb",
                        "-p",
                        "structure",
                        f"{structure}",
                        "-p",
                        "n_molecular_orbitals",
                        f"{n_molecular_orbitals}",
                        "-p",
                        "use_parallel",
                        f"{use_parallel}",
                    ]
                )


if __name__ == "__main__":
    main()
