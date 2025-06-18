from grgapp.linalg import eigs, PCs
import argparse
import pygrgl
import numpy
import json

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

    eigvals, eigvects = eigs(grg, args.num_eigs)
    PCs = PCs(grg, args.num_eigs)
    print(
        json.dumps(
            {
                "PCs": PCs.astype(numpy.float64).tolist(),
            }
        )
    )
    print(
        json.dumps(
            {
                "eigenvalues": eigvals.astype(numpy.float64).tolist(),
                # "eigenvectors": eigvects.astype(numpy.float64).tolist(),
            },
            indent=2,
        )
    )
