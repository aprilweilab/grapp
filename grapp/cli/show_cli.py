import argparse
import pygrgl
from collections import defaultdict


def add_options(subparser: argparse.ArgumentParser):
    subparser.add_argument("grg_input", help="The input GRG file")
    whattoshow = subparser.add_mutually_exclusive_group()
    whattoshow.add_argument(
        "-S",
        "--individuals",
        action="store_true",
        help="Show the individual IDs.",
    )
    whattoshow.add_argument(
        "-P",
        "--populations",
        action="store_true",
        help="Show the population information.",
    )


def run(args):
    grg = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
    print(f"Mutations: {grg.num_mutations}")
    print(f"Samples: {grg.num_samples}")
    print(f"Individuals: {grg.num_individuals}")
    print(f"Phased? {'Yes' if grg.is_phased else 'No'}")
    print(f"Nodes: {grg.num_nodes}")
    print(f"Edges: {grg.num_edges}")
    if args.individuals:
        if grg.has_individual_ids:
            print("\n".join(map(grg.get_individual_id, range(grg.num_individuals))))
    if args.populations:
        pop_labels = grg.get_populations()
        if pop_labels:
            by_pop = defaultdict(int)
            for sample_id in range(grg.num_samples):
                by_pop[grg.get_population_id(sample_id)] += 1
            for p, label in enumerate(pop_labels):
                print(f"{label}: {by_pop[p]}/{grg.num_samples} haplotypes")
