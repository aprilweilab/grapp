from grg_pheno_sim.phenotype import sim_phenotypes, convert_to_phen
from grapp.cli.util import load_immutable
import argparse
import os


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "-e",
        "--heritability",
        default=0.33,
        type=float,
        help="The heritability (h^2) of the simulated trait. Default: 0.33.",
    )
    subparser.add_argument(
        "-n",
        "--num-causal",
        type=int,
        default=None,
        help="Number of causal variants to simulate. Default: every variant is causal.",
    )
    subparser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed. Default: 42",
    )
    subparser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize the final phenotype values.",
    )
    subparser.add_argument(
        "--save-effects",
        help="Write the per-SNP effect sizes to the given filename.",
    )
    subparser.add_argument(
        "-o",
        "--out-file",
        default=None,
        help="Tab-separated output file (with header); exported Pandas DataFrame. Default: <grg_input>.phen",
    )


def run(args):
    base = os.path.basename(args.grg_input)
    output_path = f"{base}.phen"
    grg = load_immutable(args.grg_input, load_up_edges=False)
    phenotypes = sim_phenotypes(
        grg,
        num_causal=args.num_causal,
        random_seed=args.seed,
        normalize_phenotype=not args.no_normalize,
        normalize_genetic_values_before_noise=not args.no_normalize,
        heritability=args.heritability,
        save_effect_output=bool(args.save_effects is not None),
        effect_path=args.save_effects,
        header=True,
    )
    if args.out_file is None:
        args.out_file = output_path
    convert_to_phen(phenotypes, args.out_file, include_header=True)

    print()
    print(f"Wrote phenotypes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
