"""
Site-frequency-spectrum calculations from data stored in a GRG.

.. note::
    This module uses scikit-allel. This integration is optional, so "pip install grapp"
    doesn't install scikit-allel, you need to "pip install scikit-allel" yourself.
"""

try:
    import allel
except ImportError:
    allel = None  # type: ignore
import numpy
import pygrgl
from grapp.popgen.simple import (
    allele_counts,
    pop_allele_counts,
    population_pairs,
)
from typing import List, Tuple, Any


def _check_allel():
    assert (
        allel is not None
    ), "Requires 'allel' module: 'pip install scikit-allel' or 'pip install grapp[popgen]"


def _single_apply(grg: pygrgl.GRG, folded: bool, sfs_func: Any) -> numpy.typing.NDArray:
    ac = allele_counts(grg, return_ref=folded)
    assert ac.ndim == (2 if folded else 1)
    return sfs_func(ac)


def sfs(grg: pygrgl.GRG):
    _check_allel()
    return _single_apply(grg, False, allel.sfs)


def sfs_scaled(grg: pygrgl.GRG):
    _check_allel()
    return _single_apply(grg, False, allel.sfs_scaled)


def sfs_folded(grg: pygrgl.GRG):
    _check_allel()
    return _single_apply(grg, True, allel.sfs_folded)


def sfs_folded_scaled(grg: pygrgl.GRG):
    _check_allel()
    return _single_apply(grg, True, allel.sfs_folded_scaled)


# Compute all joint SFSs in the GRG, given a particular pairwise-SFS calculation function
def _joint_apply(
    grg: pygrgl.GRG, folded: bool, sfs_func: Any
) -> List[Tuple[int, int, numpy.typing.NDArray]]:
    _check_allel()
    pop_ac = pop_allele_counts(grg, return_ref=folded)
    assert pop_ac.ndim == (3 if folded else 2)
    result = []
    for pop1, pop2 in population_pairs(grg):
        result.append((pop1, pop2, sfs_func(pop_ac[pop1], pop_ac[pop2])))
    return result


def joint_sfs(grg: pygrgl.GRG) -> List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    Return all joint SFSs between all (unique) population pairs in the GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: List of ``(pop1, pop2, jSFS)`` where ``pop1``, ``pop2`` are population indices
        (see ``GRG.get_populations()``). ``jSFS`` is the :math:`\\frac{n1}{2} \\times \\frac{n2}{2}``
        matrix as emitted by ``allel.joint_sfs_folded``.
    :rtype: List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    return _joint_apply(grg, False, allel.joint_sfs)


def joint_sfs_scaled(grg: pygrgl.GRG) -> List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    Return all scaled joint SFSs between all (unique) population pairs in the GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: List of ``(pop1, pop2, jSFS)`` where ``pop1``, ``pop2`` are population indices
        (see ``GRG.get_populations()``). ``jSFS`` is the :math:`\\frac{n1}{2} \\times \\frac{n2}{2}``
        matrix as emitted by ``allel.joint_sfs_folded``.
    :rtype: List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    return _joint_apply(grg, False, allel.joint_sfs_scaled)


def joint_sfs_folded(grg: pygrgl.GRG) -> List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    Return all joint folded SFSs between all (unique) population pairs in the GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: List of ``(pop1, pop2, jSFS)`` where ``pop1``, ``pop2`` are population indices
        (see ``GRG.get_populations()``). ``jSFS`` is the :math:`\\frac{n1}{2} \\times \\frac{n2}{2}``
        matrix as emitted by ``allel.joint_sfs_folded``.
    :rtype: List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    return _joint_apply(grg, True, allel.joint_sfs_folded)


def joint_sfs_folded_scaled(
    grg: pygrgl.GRG,
) -> List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    Return all scaled joint folded SFSs between all (unique) population pairs in the GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: List of ``(pop1, pop2, jSFS)`` where ``pop1``, ``pop2`` are population indices
        (see ``GRG.get_populations()``). ``jSFS`` is the :math:`\\frac{n1}{2} \\times \\frac{n2}{2}``
        matrix as emitted by ``allel.joint_sfs_folded``.
    :rtype: List[Tuple[int, int, numpy.typing.NDArray]]:
    """
    return _joint_apply(grg, True, allel.joint_sfs_folded_scaled)
