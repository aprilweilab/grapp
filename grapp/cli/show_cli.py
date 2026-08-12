import argparse
import itertools
import numpy
import pandas
import sys
from collections import defaultdict
from grapp.util.simple import (
    allele_counts,
    allele_frequencies,
    hwe_df,
    multi_allelic_muts,
)
from grapp.cli.util import load_immutable


def add_options(subparser: argparse.ArgumentParser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of parallel jobs (threads) to use. Default: 1",
    )
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
    whattoshow.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Show the high-level GRG information.",
    )
    whattoshow.add_argument(
        "-f",
        "--frequencies",
        action="store_true",
        help="Show the tab-separated allele frequency information.",
    )
    whattoshow.add_argument(
        "-c",
        "--counts",
        action="store_true",
        help="Show the tab-separated allele counts information.",
    )
    whattoshow.add_argument(
        "-H",
        "--HWE",
        action="store_true",
        help="Show the tab-separated hardy-weinberg p-value information.",
    )
    whattoshow.add_argument(
        "--multi-sites",
        action="store_true",
        help="Show the multi-allelic sites.",
    )
    subparser.add_argument(
        "--multi-ref",
        action="store_true",
        help="Compute REF values for multi-allelic sites for anything needing zygosities (HWE, etc.).",
    )


def run(args):
    grg = load_immutable(args.grg_input, load_up_edges=False)
    if args.info:
        print(f"Ranges: mutations={grg.bp_range}, specified={grg.specified_bp_range}")
        print(f"Mutations: {grg.num_mutations}")
        print(f"Samples: {grg.num_samples}")
        print(f"Individuals: {grg.num_individuals}")
        print(f"Phased? {'Yes' if grg.is_phased else 'No'}")
        print(f"Ploidy: {grg.ploidy}")
        print(f"Nodes: {grg.num_nodes}")
        print(f"Edges: {grg.num_edges}")
        print(f"Has missing data?: {'Yes' if grg.has_missing_data else 'No'}")
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
    if args.frequencies:
        freq = allele_frequencies(grg, adjust_missing=True)
        muts = list(map(lambda m: grg.get_mutation_by_id(m), range(grg.num_mutations)))
        df = pandas.DataFrame(
            {
                "Position": numpy.array(map(lambda m: m.position, muts)),
                "REF": numpy.array(map(lambda m: m.ref_allele, muts)),
                "ALT": numpy.array(map(lambda m: m.allele, muts)),
                "Frequency": freq,
            }
        )
        df.to_csv(sys.stdout, sep="\t", index=False)
    if args.counts:
        acount, amiss = allele_counts(grg, return_missing=True)
        muts = list(map(lambda m: grg.get_mutation_by_id(m), range(grg.num_mutations)))
        df = pandas.DataFrame(
            {
                "Position": numpy.array(map(lambda m: m.position, muts)),
                "REF": numpy.array(map(lambda m: m.ref_allele, muts)),
                "ALT": numpy.array(map(lambda m: m.allele, muts)),
                "ALT Count": acount,
                "Missing Count": amiss,
                "Total": grg.num_samples,
            }
        )
        df.to_csv(sys.stdout, sep="\t", index=False)
    if args.HWE:
        df = hwe_df(grg, jobs=args.jobs, show_progress=True, all_multi=args.multi_ref)
        df.to_csv(sys.stdout, sep="\t", index=False)
    if args.multi_sites:
        multi = multi_allelic_muts(grg)
        muts = list(
            map(
                lambda m: grg.get_mutation_by_id(m),
                itertools.chain.from_iterable(multi),
            )
        )
        df = pandas.DataFrame(
            {
                "Position": numpy.array(list(map(lambda m: m.position, muts))),
                "REF": numpy.array(list(map(lambda m: m.ref_allele, muts))),
                "ALT": numpy.array(list(map(lambda m: m.allele, muts))),
            }
        )
        df.to_csv(sys.stdout, sep="\t", index=False)
