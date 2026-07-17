from grapp.util.simple import allele_frequencies
from grapp.grg_calculator import GRGCalculator, GRGCalcInterface, GRGSpMVCalculator
from typing import Optional, List, Union
import glob
import itertools
import numpy
import os
import pygrgl
import shutil
import sys
import subprocess

try:
    import pygrgl_spmv as _pygrgl_spmv
except ImportError:
    _pygrgl_spmv = None

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")
REPO_ROOT = os.path.join(THIS_DIR, "..")


def construct_grg(
    input_file: str,
    output_file: Optional[str] = None,
    jobs: int = 6,
    is_test_input: bool = True,
    ignore_missing: bool = False,
) -> str:
    cmd = [
        "grg",
        "construct",
        "--force",
        "-p",
        "10",
        "-j",
        str(jobs),
        os.path.join(INPUT_DIR, input_file) if is_test_input else input_file,
    ]
    if ignore_missing:
        cmd.append("--ignore-missing")
    if output_file is not None:
        cmd.extend(["-o", output_file])
    else:
        output_file = os.path.basename(input_file) + ".final.grg"
    subprocess.check_call(cmd)
    return output_file


# It is important that this only uses down edges, so that we can test against
# matmul methods which (should) only require down edges.
def grg2X(grg: pygrgl.GRG, diploid: bool = False):
    samples = grg.num_individuals if diploid else grg.num_samples
    result = numpy.zeros((samples, grg.num_mutations))
    samps_below = [list() for _ in range(grg.num_nodes)]
    for node_id in range(grg.num_nodes):
        sb = []
        if grg.is_sample(node_id):
            sb.append(node_id)
        for child_id in grg.get_down_edges(node_id):
            sb.extend(samps_below[child_id])
        samps_below[node_id] = sb

        muts = grg.get_mutations_for_node(node_id)
        if muts:
            for sample_id in sb:
                indiv = sample_id // grg.ploidy
                for mut_id in muts:
                    if diploid:
                        result[indiv][mut_id] += 1
                    else:
                        result[sample_id][mut_id] = 1
    # Handle missingness afterwards for simplicity (harder to make mistakes this way). Each cell for
    # a missing datum is filled in with f_i for mutation i (haplotypes) or 2*f_i (diploid matrix).
    if grg.has_missing_data:
        freqs = allele_frequencies(grg, adjust_missing=True)
        for mut_id, mut_node, miss_node in grg.get_mutation_node_miss():
            if miss_node != pygrgl.INVALID_NODE:
                for sample_id in samps_below[miss_node]:
                    if diploid:
                        indiv = sample_id // grg.ploidy
                        result[indiv][mut_id] += freqs[mut_id]
                    else:
                        assert (
                            result[sample_id][mut_id] == 0
                        ), f"{result[sample_id][mut_id]}"
                        result[sample_id][mut_id] = freqs[mut_id]
    return result


def standardize_X(X: numpy.ndarray):
    """
    X: N×M diploid genotype matrix with entries in {0,1,2}.
    Returns:
      Xstd: N×M standardized matrix,
      freqs: length-M allele freqs f_i,
      sigma: length-M stddev sqrt(2 f_i (1-f_i))
    """
    N, M = X.shape
    # allele frequency per variant
    freqs = X.sum(axis=0) / (2 * N)
    # U is N×M each column = 2 f_i
    U = 2 * freqs
    # center
    Xc = X - U[None, :]
    # s_i = sqrt(2 f_i (1-f_i))
    sigma = numpy.sqrt(2 * freqs * (1 - freqs))
    # avoid division by zero
    zero_sigma = sigma == 0
    if numpy.any(zero_sigma):
        print(
            f"Warning: {zero_sigma.sum()} sites are monomorphic → will stay zero after std."
        )
        sigma[zero_sigma] = 1.0
    # standardize
    Xstd = Xc / sigma[None, :]
    # re-zero freq 1 cols
    Xstd[:, zero_sigma] = 0
    return Xstd


# Split a GRG and load all the parts, and returns them _sorted by position_
def split_and_load(
    grg_filename: str,
    out_dir: str,
    size_per: int,
    jobs: int,
    cleanup: bool = True,
    filenames: Optional[List] = None,
):
    subprocess.check_output(
        [
            "grg",
            "split",
            "-j",
            str(jobs),
            grg_filename,
            "-s",
            str(size_per),
            "-o",
            out_dir,
        ]
    )
    grgs = []
    for fn in glob.glob(os.path.join(out_dir, "*.grg")):
        grgs.append(pygrgl.load_immutable_grg(fn))
        if filenames is not None:
            filenames.append(fn)
    grgs.sort(key=lambda g: g.bp_range[0])
    if cleanup:
        shutil.rmtree(out_dir)
    return grgs


# Returns four lists: keep_indivs, ignore_indivs, keep_samples, ignore_samples, which are
# all consistent.
def complete_sample_sets(grg, ignore_indivs):
    keep_indivs = [i for i in range(grg.num_individuals) if i not in ignore_indivs]
    keep_samples = list(
        itertools.chain.from_iterable(map(lambda i: (2 * i, 2 * i + 1), keep_indivs))
    )
    ignore_samples = [i for i in range(grg.num_samples) if i not in keep_samples]
    assert set(keep_samples) | set(ignore_samples) == set(range(grg.num_samples))
    return keep_indivs, ignore_indivs, keep_samples, ignore_samples


def _wrap_grg_spmv(grg: Union[pygrgl.GRG, GRGCalcInterface]) -> GRGCalcInterface:
    _SPMV_ENV = "PYGRGL_SPMV_CONFIG"
    config_path = os.environ.get(_SPMV_ENV)
    if config_path is None:
        print(
            f"[_wrap_grg_spmv] Environment variable {_SPMV_ENV!r} is NOT set.\n"
            f"  pygrgl_spmv.load() will attempt to select a default backend "
            f"(MKL preferred, cuSPARSE fallback) and emit a UserWarning.\n"
            f"  To silence this warning and control the backend, set {_SPMV_ENV} "
            f"to the path of a JSON config file."
        )
    else:
        print(
            f"[_wrap_grg_spmv] Environment variable {_SPMV_ENV!r} is set to: {config_path!r}"
        )

    if isinstance(grg, GRGCalcInterface):
        print(
            f"[_wrap_grg_spmv] Input is already a GRGCalcInterface ({type(grg).__name__}); returning as-is."
        )
        return grg

    if _pygrgl_spmv is None:
        raise ImportError(
            "pygrgl_spmv is not installed; cannot use _wrap_grg_spmv. "
            "Install the grg-spmv package or use _wrap_grg instead."
        )
    print(
        f"[_wrap_grg_spmv] Loading GRG object ({type(grg).__name__}) via pygrgl_spmv.load() ..."
    )
    spmv_op = _pygrgl_spmv.load(grg)
    print(f"[_wrap_grg_spmv] Load complete. Wrapping in GRGSpMVCalculator.")
    return GRGSpMVCalculator(spmv_op)


WRAP_GRG_PARAMS = [(lambda g: g,), (GRGCalculator,)]
if _pygrgl_spmv is not None:
    WRAP_GRG_PARAMS.append((_wrap_grg_spmv,))


def which(exe: str, required=False) -> Optional[str]:
    """
    Find an executable on $PATH or in the python system path.
    """
    try:
        result = (
            subprocess.check_output(["which", exe], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        result = None
    if result is None:
        for p in [*sys.path, REPO_ROOT]:
            p = os.path.join(p, exe)
            if os.path.isfile(p):
                result = p
                break
    if required and result is None:
        raise RuntimeError(f"Could not find executable {exe}")
    return result


MAIN_PY = os.path.join(REPO_ROOT, "grapp", "cli", "main.py")


def grapp_run(*command) -> str:
    return subprocess.check_output(
        ["python", MAIN_PY] + list(command), stderr=subprocess.STDOUT
    ).decode("utf-8")


def make_grg_sparse_mat(
    positions: List[int],
    ref_alleles: List[str],
    alt_alleles: List[str],
    matrix: numpy.typing.NDArray,
    miss: Optional[numpy.typing.NDArray] = None,
    ploidy: int = 1,
) -> pygrgl.GRG:
    """
    Create a GRG that is equivalent to the sparse matrix representations of the given genotype
    matrix. I.e., this is not a GRG that compresses the data, but just a naive graph representation
    of the mapping between mutations and samples.
    """
    N = matrix.shape[0]
    M = len(positions)
    assert M == len(ref_alleles)
    assert M == len(alt_alleles)
    assert matrix.shape[1] == M
    g = pygrgl.MutableGRG(N, ploidy)

    def add_samples(sample_vector):
        # Create the mutation node
        node = g.make_node()
        for j in numpy.flatnonzero(sample_vector):
            g.connect(node, j)
        if ploidy == 2:
            dosage = sample_vector[0::ploidy] + sample_vector[1::ploidy]
            coals = numpy.count_nonzero(dosage == ploidy)
            g.set_num_individual_coals(node, coals)
        return node

    pos2miss = {}
    for i in range(M):
        sample_vector = matrix[:, i]
        mut_node = add_samples(sample_vector)
        pos = positions[i]
        miss_node = pygrgl.INVALID_NODE
        if pos in pos2miss:
            miss_node = pos2miss[pos]
        elif miss is not None:
            miss_vector = miss[:, i]
            if numpy.sum(miss_vector) > 0:
                miss_node = add_samples(miss_vector)
        pos2miss[pos] = miss_node
        g.add_mutation(
            pygrgl.Mutation(positions[i], alt_alleles[i], ref_alleles[i]),
            mut_node,
            miss_node,
        )
    return g
