from typing import Optional
import numpy
import os
import pygrgl
import subprocess

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


def construct_grg(
    input_file: str,
    output_file: Optional[str] = None,
    jobs: int = 6,
    is_test_input: bool = True,
) -> str:
    cmd = [
        "grg",
        "construct",
        "-p",
        "10",
        "-t",
        "2",
        "-j",
        str(jobs),
        os.path.join(INPUT_DIR, input_file) if is_test_input else input_file,
    ]
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
