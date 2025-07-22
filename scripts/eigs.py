from grgapp.linalg import eigs, PCs
import argparse
import pygrgl
import numpy
import json
import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grg", help="GRG input file")
    parser.add_argument(
        "-d",
        "--num-eigs",
        type=int,
        default=10,
        help="Number of eigenvalues/vectors to compute",
    )
    args = parser.parse_args()

    grg = pygrgl.load_immutable_grg(args.grg)

    # eigvals, eigvects = eigs(grg, args.num_eigs)
    # t0 = time.perf_counter()
    PCs = PCs(grg, args.num_eigs,False)
    # t1 = time.perf_counter()
    # print(t1-t0)
    print(PCs)
    # print(
    #     json.dumps(
    #         {
    #             "eigenvalues": eigvals.astype(numpy.float64).tolist(),
    #             # "eigenvectors": eigvects.astype(numpy.float64).tolist(),
    #         },
    #         indent=2,
    #     )
    # )
