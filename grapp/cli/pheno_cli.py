from grg_pheno_sim.phenotype import sim_phenotypes
import argparse
import os
import pygrgl


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
        action="store_true",
        help="Write file <grg_input>.effects.par, containing the per-SNP effect sizes.",
    )


def run(args):
    base = os.path.basename(args.grg_input)
    effect_path = f"{base}.effects.par"
    output_path = f"{base}.phen"
    grg = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
    sim_phenotypes(
        grg,
        num_causal=args.num_causal,
        random_seed=args.seed,
        normalize_phenotype=not args.no_normalize,
        normalize_genetic_values_before_noise=True,
        heritability=args.heritability,
        save_effect_output=args.save_effects,
        effect_path=effect_path,
        standardized_output=True,
        path=output_path,
        header=True,
    )
    print()
    print(f"Wrote phenotypes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
