from contextlib import contextmanager
from grapp.cli.util import load_immutable
from grapp.util.filter import split_by_ranges
from multiprocessing import Pool
from typing import Optional, Callable, Union, Any, List, Dict
import os
import pygrgl
import sys
import tempfile


# Helper to make the usage of a user-specified directory and a temporary directory seemless.
def _get_temp_dir_context(temp_dir: Optional[str] = None) -> Callable:
    if temp_dir is None:
        return tempfile.TemporaryDirectory

    @contextmanager
    def existing_dir_context_mgr(*args, **kwargs):
        yield temp_dir

    return existing_dir_context_mgr


def split_and_run(
    grg_or_filename: Union[pygrgl.GRG, str],
    operation: Callable[[Union[str, pygrgl.GRG], Dict[str, Any]], Any],
    merge_operation: Callable[
        [List[Union[str, pygrgl.GRG]], List[Any], Dict[str, Any]], Any
    ],
    context: Dict[str, Any],
    jobs: int = 1,
    temp_dir: Optional[str] = None,
    split_threshold: int = 1_000_000,
    verbose: bool = False,
) -> Any:
    """
    Perform an arbitrary GRG operation in parallel by splitting the GRG into smaller graphs,
    running the operation on each subgraph, and then merging the results. This can
    be used for mutable or immutable operations. For a mutable operation, the GRGs
    will be merged into a final GRG, with the filename given.

    The context dictionary that is passed between callback functions will always contain
    the "dir" key, which is the directory that the splitting and running is occurring in,
    and is where any intermediate result files (temp files) should be placed by the operation.

    :param grg_or_filename: The GRG to convert, either as a pygrgl.GRG or the
        filename of a GRG.
    :type grg: Union[pygrgl.GRG, str]
    :param operation: Function that takes (GRG, context_dict) and returns a result (of any type).
        The GRG is either a string or a pygrgl.GRG object, and this operation is performed on that
        GRG after it was split out from the larger GRG. Use context_dict to pass information to the
        operation if needed. The result from this operation will be collected in a list, and that
        list will be passed to the merge_operation.
    :type operation: Callable[[Union[str, pygrgl.GRG], Dict[str, Any]], Any],
    :param merge_operation: Function that takes (list(GRG), list(results), context_dict) and returns
        a result (of any type). The list(results) are all the return values from operation. The list of
        GRGs are all the GRGs that the operation was run on. The context dictionary can be used to pass
        information to the merge operation.
    :type merge_operation: Callable[[List[Union[str, pygrgl.GRG]], List[Any], Dict[str, Any]], Any]
    :param context: The context dictionary, can be empty, or contain any information that you want to
        pass to both callback functions. The key "dir" is reserved (see above).
    :type context: Dict[str, Any]
    :param jobs: The number of parallel processes to use to do the conversion. The
        speed-up is essentially linear. Default: 1.
    :type jobs: int
    :param temp_dir: The directory to use for intermediate IGD files. The GRG
        is split into multiple pieces and placed in this directory, and then each
        piece gets converted to an IGD file, and then those IGD files are merged
        into the final result. If temp_dir is None, these files are placed in a
        temporary directory which is then deleted upon completion.
    :type temp_dir: Optional[str]
    :param split_threshold: Basepair threshold for splitting the GRG into chunks
        for processing. A split GRG can be much faster to operate on than a full sized
        GRG, plus this is how we parallelize the operation. Default: 5MB.
    :type split_threshold: int
    """
    if verbose:

        def logv(msg):
            print(msg, file=sys.stderr)

    assert temp_dir is None or os.path.isdir(
        temp_dir
    ), f"Provided temp_dir {temp_dir} does not exist."

    with _get_temp_dir_context(temp_dir)() as tmpdirname:
        if isinstance(grg_or_filename, str):
            grg = load_immutable(grg_or_filename, load_up_edges=False)
            grg_filename = grg_or_filename
        else:
            grg = grg_or_filename
            grg_filename = os.path.join(tmpdirname, "input.grg")
            pygrgl.save_grg(grg, grg_filename)

        # Add the temporary directory to the context for operations to use.
        context["dir"] = tmpdirname

        split_ranges = []
        for start in range(grg.bp_range[0], grg.bp_range[1], split_threshold):
            split_ranges.append((start, start + split_threshold))
        if len(split_ranges) == 1:
            logv(f"Running on a single GRG part ...")
            single_result = operation(grg, context)
            result = merge_operation([grg], [single_result], context)
        else:
            logv(f"Using temporary directory {tmpdirname}.")
            logv(f"Splitting GRG into {len(split_ranges)} parts..")
            grg_parts = split_by_ranges(
                grg_filename, split_ranges, jobs, out_dir=tmpdirname
            )
            arguments = [(part, context) for part in filter(os.path.isfile, grg_parts)]
            assert len(arguments) > 0, "FAILURE: No GRG parts found"
            logv("Performing operation on GRG parts...")
            with Pool(jobs) as pool:
                part_results = pool.starmap(operation, arguments)
            logv(f"Merging {len(part_results)} parts into single result...")
            result = merge_operation(grg_parts, part_results, context)
    return result
