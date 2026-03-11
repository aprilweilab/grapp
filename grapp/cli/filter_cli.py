from grapp.util.filter import (
    grg_save_individuals,
    grg_save_mut_filter,
    grg_save_populations,
    grg_save_samples,
)
from grapp.util.simple import (
    UserInputError,
    allele_counts,
    allele_frequencies,
    VariantType,
    get_variant_type,
)
from typing import Set
import argparse
import os
import pygrgl


def variant_type_set(arg_value: str) -> Set[VariantType]:
    choices = set(map(lambda v: v.value, VariantType))
    result = set()
    for typ in arg_value.split(","):
        if typ not in choices:
            raise UserInputError(f"{typ} is not a valid type (try one of {choices})")
        result.add(VariantType[typ.upper()])
    assert len(result) > 0
    return result


def list_or_filename(arg_value):
    parts = arg_value.split(",")
    if len(parts) == 1:
        if not os.path.isfile(arg_value):
            raise UserInputError(f"Cannot access file {arg_value}")
        with open(arg_value) as f:
            return list(map(str.strip, f))
    return list(map(str.strip, parts))


def int_list_or_filename(arg_value):
    try:
        return list(map(int, list_or_filename(arg_value)))
    except ValueError:
        raise UserInputError("One or more argument values were not integers")


def genome_range(arg_value):
    parts = arg_value.split("-")
    if len(parts) != 2:
        raise UserInputError(f"Invalid range specification: {arg_value}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise UserInputError(f"Range specification must use integers: {arg_value}")


def add_options(subparser: argparse.ArgumentParser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument("grg_output", help="The output GRG file")
    sample_group = subparser.add_argument_group("sample filters")
    filters = sample_group.add_mutually_exclusive_group()
    # TODO: update the pygrgl APIs to take a filename _or_ a file object, so that we can do stdout piping
    filters.add_argument(
        "-S",
        "--individuals",
        type=list_or_filename,
        help="Keep only the individuals with the IDs given as a comma-separated list or in the given filename.",
    )
    filters.add_argument(
        "--hap-samples",
        type=int_list_or_filename,
        help="Keep only the haploid samples with the NodeIDs (indexes) given as a comma-separated list or in the given filename.",
    )
    filters.add_argument(
        "-P",
        "--populations",
        type=list_or_filename,
        help="Keep only the individuals with populations matching the comma-separated list or in the given filename.",
    )

    # You can specify many different filters on mutations simultaneously, but (above) you cannot specify both
    # mutation and sample based filters.
    mutation_group = subparser.add_argument_group("mutation filters")
    mutation_group.add_argument(
        "-r",
        "--range",
        type=genome_range,
        help='Keep only the variants within the given range, in base pairs. Example: "lower-upper", where both are integers '
        "and lower is inclusive, upper is exclusive.",
    )
    mutation_group.add_argument(
        "-c",
        "--min-ac",
        type=int,
        help="Minimum allele count to keep. All Mutations with count below this value will be dropped",
    )
    mutation_group.add_argument(
        "-C",
        "--max-ac",
        type=int,
        help="Maximum allele count to keep. All Mutations with count above this value will be dropped",
    )
    mutation_group.add_argument(
        "-q",
        "--min-af",
        type=float,
        help="Minimum allele frequency to keep. All Mutations with frequency below this value will be dropped",
    )
    mutation_group.add_argument(
        "-Q",
        "--max-af",
        type=float,
        help="Maximum allele frequency to keep. All Mutations with frequency above this value will be dropped",
    )
    mutation_group.add_argument(
        "-v",
        "--types",
        type=variant_type_set,
        help="Comma-separated list of variant types to select. Site is selected if any of the ALT alleles is of the type requested. Types are determined by comparing the REF and ALT alleles.",
    )
    mutation_group.add_argument(
        "-A",
        "--apply-to-sites",
        action="store_true",
        help="By default, all filters apply to each variant independently. This flag will cause an entire site to be dropped if any variants are filtered out.",
    )
    mutation_group.add_argument(
        "-m",
        "--min-alleles",
        type=int,
        default=None,
        help="Only keep sites with at least this many alleles. Counts all REF alleles as 1.",
    )
    mutation_group.add_argument(
        "-M",
        "--max-alleles",
        type=int,
        default=None,
        help="Only keep sites with at most this many alleles. Counts all REF alleles as 1. "
        "Use '-m 2 -M 2 -v snps' to view only biallelic SNPs.",
    )


def require_unspecified(args, msg, *params):
    for argname in params:
        if getattr(args, argname) is not None:
            raise RuntimeError(msg)


def run(args):
    def no_mut_filters():
        require_unspecified(
            args,
            "Cannot mix sample and mutation filters",
            "range",
            "min_ac",
            "max_ac",
            "min_af",
            "max_af",
            "types",
            "min_alleles",
            "max_alleles",
        )

    if args.individuals:
        no_mut_filters()
        grg_save_individuals(
            args.grg_input, args.grg_output, args.individuals, verbose=True
        )
    elif args.hap_samples:
        no_mut_filters()
        grg_save_samples(
            args.grg_input, args.grg_output, args.hap_samples, verbose=True
        )
    elif args.populations:
        no_mut_filters()
        grg_save_populations(
            args.grg_input, args.grg_output, args.populations, verbose=True
        )
    else:
        if args.min_alleles is None:
            args.min_alleles = 1
        if args.max_alleles is None:
            args.max_alleles = 2**32
        if args.range is None:
            brange = (0, 0)
        else:
            brange = args.range
        grg = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
        counts = None
        freqs = None
        if args.min_ac is not None or args.max_ac is not None:
            counts = allele_counts(grg)
        if args.min_af is not None or args.max_af is not None:
            freqs = allele_frequencies(grg)

        def filter_method(grg: pygrgl.GRG, mut_id: int):
            mut = grg.get_mutation_by_id(mut_id)
            if args.range is not None and not (
                mut.position >= brange[0] and mut.position < brange[1]
            ):
                return False
            if args.min_ac is not None and counts[mut_id] < args.min_ac:
                return False
            if args.max_ac is not None and counts[mut_id] > args.max_ac:
                return False
            if args.min_af is not None and freqs[mut_id] < args.min_af:
                return False
            if args.max_af is not None and freqs[mut_id] > args.max_af:
                return False
            if args.types is not None:
                my_type = get_variant_type(mut)
                return my_type in args.types
            return True

        kept, dropped = grg_save_mut_filter(
            args.grg_input,
            args.grg_output,
            filter_method,
            brange,
            apply_to_sites=args.apply_to_sites,
            min_variants=args.min_alleles - 1,  # Counting the REF allele
            max_variants=args.max_alleles - 1,  # Counting the REF allele
        )
        print(f"Kept {kept} variants. Dropped {dropped} variants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
