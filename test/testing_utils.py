from typing import Optional
import numpy
import os
import pygrgl
import subprocess

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


def construct_grg(
    input_file: str, output_file: Optional[str] = None, jobs: int = 6
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
        os.path.join(INPUT_DIR, input_file),
    ]
    if output_file is not None:
        cmd.extend(["-o", output_file])
    else:
        output_file = os.path.basename(input_file) + ".final.grg"
    subprocess.check_call(cmd)
    return output_file


def grg2X(grg: pygrgl.GRG, diploid: bool = False):
    samples = grg.num_individuals if diploid else grg.num_samples
    ploidy = grg.ploidy
    result = numpy.zeros((samples, grg.num_mutations))
    muts_above = {}
    for node_id in reversed(range(grg.num_nodes)):
        muts = grg.get_mutations_for_node(node_id)
        ma = []
        if muts:
            ma.extend(muts)
        for parent_id in grg.get_up_edges(node_id):
            ma.extend(muts_above[parent_id])
        muts_above[node_id] = ma
        if grg.is_sample(node_id):
            indiv = node_id // ploidy
            for mut_id in muts_above[node_id]:
                if diploid:
                    result[indiv][mut_id] += 1
                else:
                    result[node_id][mut_id] = 1
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
