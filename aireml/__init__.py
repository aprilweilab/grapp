"""
aireml: average-information REML for operator-defined variance components.

A small, self-contained implementation of the AI-REML algorithm (Gilmour
et al. 1995) in the matrix-free form described by Lee et al. (2026), "Genetic
prediction with ARG-powered linear algebra" (GENETICS 233(1), iyag074).

The only thing the solver needs from the relatedness structure is a
matrix-vector product, so the genetic covariance is supplied as a
``scipy.sparse.linalg.LinearOperator``.  That makes the package independent of
how relatedness is represented: a dense GRM, a genotype matrix, an ARG, or a
GRG all work as long as they can multiply a vector.

    >>> import numpy
    >>> from aireml import fit_reml
    >>> rng = numpy.random.default_rng(0)
    >>> genotypes = rng.normal(size=(500, 2000))
    >>> from aireml import grm_from_genotypes
    >>> grm = grm_from_genotypes(genotypes)
    >>> y = rng.multivariate_normal(numpy.zeros(500), numpy.eye(500))
    >>> result = fit_reml(y, grm, seed=1)          # doctest: +SKIP
    >>> result.heritability                        # doctest: +SKIP

The package depends only on numpy and scipy.
"""

from .operators import (
    as_symmetric_operator,
    centered_operator,
    grm_from_genotypes,
    identity_operator,
    LinearCombinationOperator,
    materialize,
)
from .model import RemlState
from .reml import fit_reml, haseman_elston, REMLResult
from .solvers import (
    ConjugateGradientSolver,
    CovarianceSolver,
    DenseSolver,
    NystromSketch,
)
from .trace import estimate_trace, TraceEstimate

__version__ = "0.1.0"

__all__ = [
    "as_symmetric_operator",
    "centered_operator",
    "ConjugateGradientSolver",
    "CovarianceSolver",
    "DenseSolver",
    "estimate_trace",
    "fit_reml",
    "grm_from_genotypes",
    "haseman_elston",
    "identity_operator",
    "LinearCombinationOperator",
    "materialize",
    "NystromSketch",
    "REMLResult",
    "RemlState",
    "TraceEstimate",
    "__version__",
]
