from grgapp.assoc import (
    read_covariates_matrix, read_pheno, linear_assoc_covar, linear_assoc_no_covar,
)
from pstats import SortKey
import argparse
import cProfile, pstats, io
import pygrgl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grg_file", help="GRG file to load")
    parser.add_argument("phenotypes", help="Phenotype text file to load")
    parser.add_argument("-c", "--covariates", help="Covariates text file to load")
    parser.add_argument("-s", "--show", default=100, help="Show the first X results (default: 100)")
    args = parser.parse_args()

    if args.covariates is not None:
        y = read_pheno(args.phenotypes)
        g = pygrgl.load_immutable_grg(args.grg_file)
        C = read_covariates_matrix(args.covariates, True)

        pr = cProfile.Profile()
        pr.enable()
        df = linear_assoc_covar(g,y,C)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        print(df.head(100))

    else:
        y = read_pheno(args.phenotypes)
        g = pygrgl.load_immutable_grg(args.grg_file)
        df = linear_assoc_no_covar(g,y)
        print(df.head(100))
