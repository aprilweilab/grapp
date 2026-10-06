"""
Runtime and accuracy benchmark for the standalone ``aireml`` package.

The point of a matrix-free AI-REML is that its cost is driven by the number of
products against the relatedness operator, not by ``N^2`` storage or ``N^3``
factorization.  This script measures that on synthetic data where the
relatedness operator is a sparse genotype matrix (a stand-in for any operator
with a cheap product -- an ARG, a GRG, a memory-mapped genotype file).

Usage::

    python examples/aireml_benchmark.py                # everything, small
    python examples/aireml_benchmark.py --sizes 1000 2000 4000
    python examples/aireml_benchmark.py --skip-accuracy
"""

import argparse
import time

import numpy
import scipy.sparse

from aireml import fit_reml, grm_from_genotypes, haseman_elston
from aireml.operators import as_symmetric_operator, identity_operator


def simulate(size, num_variants, tau2, sigma2, density, num_groups, rng):
    """
    A sparse genotype matrix with group structure, plus a phenotype drawn from
    ``y = mu + g + e`` with ``g ~ Normal(0, tau2 * Z Z^T / M)``.

    Sampling ``g`` as ``tau * Z u / sqrt(M)`` avoids ever forming the GRM, so
    the simulation scales as well as the estimator does.
    """
    genotypes = scipy.sparse.random(
        size,
        num_variants,
        density=density,
        format="csr",
        random_state=numpy.random.RandomState(rng.integers(2**31)),
        data_rvs=lambda n: rng.integers(1, 3, size=n).astype(numpy.float64),
    )
    # Give the sample some relatedness structure: individuals in a group share
    # a block of variants, which puts a few large eigenvalues in the GRM.
    membership = rng.integers(0, num_groups, size=size)
    block = max(1, num_variants // (4 * num_groups))
    rows, cols = [], []
    for group in range(num_groups):
        members = numpy.flatnonzero(membership == group)
        start = (group * block) % max(1, num_variants - block)
        for column in range(start, start + block):
            rows.extend(members.tolist())
            cols.extend([column] * members.size)
    shared = scipy.sparse.csr_matrix(
        (numpy.ones(len(rows)), (rows, cols)), shape=(size, num_variants)
    )
    genotypes = (genotypes + shared).tocsr()

    # Normalize so the GRM has mean diagonal 1; then tau2 is a genetic
    # variance and tau2 / (tau2 + sigma2) is a heritability.  (Lee et al. 2026
    # discuss why an unnormalized branch GRM gives tau2 a different meaning.)
    scale = float(genotypes.multiply(genotypes).sum()) / size
    operator = grm_from_genotypes(genotypes, scale=scale)
    effects = rng.standard_normal(num_variants)
    genetic = genotypes @ effects / numpy.sqrt(scale)
    phenotype = (
        1.5
        + numpy.sqrt(tau2) * genetic
        + numpy.sqrt(sigma2) * rng.standard_normal(size)
    )
    return operator, phenotype, genotypes.nnz


def runtime_benchmark(args):
    print("\n=== runtime scaling ===")
    print(
        f"{'N':>8} {'nnz(Z)':>10} {'iters':>6} {'solves':>8} {'matvecs':>9} "
        f"{'seconds':>9} {'s/1e3 ind':>10} {'tau2':>8} {'sigma2':>8} {'h2':>7}"
    )
    truth = args.tau2 / (args.tau2 + args.sigma2)
    rows = []
    for size in args.sizes:
        rng = numpy.random.default_rng(args.seed)
        operator, phenotype, nnz = simulate(
            size,
            args.variants,
            args.tau2,
            args.sigma2,
            args.density,
            args.groups,
            rng,
        )
        started = time.perf_counter()
        result = fit_reml(
            phenotype,
            operator,
            num_trace_vectors=args.trace_vectors,
            preconditioner_rank=args.preconditioner_rank,
            cg_tol=args.cg_tol,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - started
        solver = result._state.solver
        print(
            f"{size:>8} {nnz:>10} {result.num_iterations:>6} "
            f"{solver.num_solves:>8} {solver.num_matvecs:>9} {elapsed:>9.2f} "
            f"{1000.0 * elapsed / size:>10.3f} "
            f"{result.variance_components[0]:>8.4f} "
            f"{result.variance_components[1]:>8.4f} {result.heritability:>7.4f}"
        )
        rows.append((size, elapsed))
    print(f"(simulated h2 = {truth:.4f})")
    if len(rows) > 1:
        sizes = numpy.array([row[0] for row in rows], dtype=float)
        times = numpy.array([row[1] for row in rows], dtype=float)
        slope = numpy.polyfit(numpy.log(sizes), numpy.log(times), 1)[0]
        print(f"empirical scaling: runtime ~ N^{slope:.2f}")


def accuracy_benchmark(args):
    """
    Compare, on problems small enough to solve exactly:

    * AI-REML with exact traces and exact solves, against a brute-force
      maximization of the REML likelihood (are the updates right?);
    * the matrix-free stochastic path, against that exact optimum (does the
      randomized linear algebra cost accuracy?);
    * Haseman-Elston, against the same target (Lee et al. 2026 report REML is
      substantially more accurate; Fig. 5).
    """
    from scipy.optimize import minimize

    from aireml.model import RemlState
    from aireml.solvers import DenseSolver

    print(
        "\n=== accuracy (N = %d, %d replicates) ==="
        % (args.accuracy_size, args.replicates)
    )
    print(
        f"{'true h2':>8} {'exact REML':>12} {'AI-REML':>10} {'matrix-free':>12} "
        f"{'HE':>10} {'MAE REML':>10} {'MAE HE':>8}"
    )
    size = args.accuracy_size
    for true_h2 in (0.2, 0.5, 0.8):
        tau2, sigma2 = true_h2, 1.0 - true_h2
        exact_estimates, aireml_estimates, free_estimates, he_estimates = [], [], [], []
        for replicate in range(args.replicates):
            rng = numpy.random.default_rng(1000 * replicate + int(100 * true_h2))
            genotypes = rng.standard_normal((size, 4 * size))
            grm = genotypes @ genotypes.T / (4 * size)
            covariates = numpy.ones((size, 1))
            covariance = tau2 * grm + sigma2 * numpy.eye(size)
            phenotype = 1.5 + numpy.linalg.cholesky(covariance) @ rng.standard_normal(
                size
            )

            operators = [as_symmetric_operator(grm), identity_operator(size)]
            state = RemlState(phenotype, covariates, DenseSolver(operators))

            def objective(log_theta):
                state.update(numpy.exp(log_theta))
                return state.restricted_objective()

            reference = numpy.exp(
                minimize(
                    objective,
                    numpy.log([0.5, 0.5]),
                    method="Nelder-Mead",
                    options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000),
                ).x
            )
            exact_estimates.append(reference[0] / reference.sum())

            fitted = fit_reml(
                phenotype,
                grm,
                solver="dense",
                trace_method="exact",
                tolerance=1e-8,
                extra_iterations=0,
                seed=replicate,
            )
            aireml_estimates.append(fitted.heritability)

            free = fit_reml(
                phenotype,
                grm_from_genotypes(genotypes),
                num_trace_vectors=args.trace_vectors,
                preconditioner_rank=args.preconditioner_rank,
                seed=replicate,
            )
            free_estimates.append(free.heritability)

            moments = haseman_elston(
                phenotype,
                operators,
                covariates,
                num_vectors=args.trace_vectors,
                rng=numpy.random.default_rng(replicate),
            )
            he_estimates.append(moments[0] / moments.sum())

        exact_estimates = numpy.array(exact_estimates)
        print(
            f"{true_h2:>8.2f} {exact_estimates.mean():>12.4f} "
            f"{numpy.mean(aireml_estimates):>10.4f} "
            f"{numpy.mean(free_estimates):>12.4f} "
            f"{numpy.mean(he_estimates):>10.4f} "
            f"{numpy.mean(numpy.abs(numpy.array(free_estimates) - true_h2)):>10.4f} "
            f"{numpy.mean(numpy.abs(numpy.array(he_estimates) - true_h2)):>8.4f}"
        )


def trace_benchmark(args):
    """
    Compare the trace estimators, which is where the runtime goes.

    Two views: the sampling spread of each estimator on a matrix whose trace
    we know exactly, and the effect of each on a whole AI-REML fit.  Both are
    reported at equal *matrix-vector* budget, because every product against
    ``P V_i`` is a conjugate-gradient solve -- XTrace spends two products per
    test vector while Hutchinson and Hutch++ spend one, so equal
    ``num_vectors`` would not be a fair comparison.
    """
    from aireml.trace import estimate_trace
    from aireml.operators import materialize
    from scipy.sparse.linalg import aslinearoperator

    budget = args.trace_budget
    print(f"\n=== trace estimator spread (equal budget of {budget} products) ===")
    rng = numpy.random.default_rng(args.seed)
    size = 400
    basis, _ = numpy.linalg.qr(rng.standard_normal((size, size)))
    matrix = basis @ numpy.diag(numpy.arange(1, size + 1) ** -1.5) @ basis.T
    operator = aslinearoperator(matrix)
    exact = numpy.trace(matrix)
    print(f"exact trace = {exact:.6f}")
    print(
        f"{'method':12s} {'num_vectors':>11} {'products':>9} {'mean':>10} "
        f"{'bias':>11} {'std':>10} {'rel.std':>8}"
    )
    baseline = None
    for method, num_vectors in (
        ("hutchinson", budget),
        ("hutchpp", budget),
        ("xtrace", budget // 2),
    ):
        values = numpy.array(
            [
                estimate_trace(
                    operator, num_vectors, method, numpy.random.default_rng(seed)
                ).value
                for seed in range(args.trace_repeats)
            ]
        )
        products = estimate_trace(
            operator, num_vectors, method, numpy.random.default_rng(0)
        ).num_matvecs
        spread = values.std(ddof=1)
        baseline = spread if baseline is None else baseline
        print(
            f"{method:12s} {num_vectors:>11} {products:>9} {values.mean():>10.6f} "
            f"{values.mean() - exact:>+11.2e} {spread:>10.3e} "
            f"{spread / baseline:>8.3f}"
        )

    print(f"\n=== effect on a whole AI-REML fit (N = {args.trace_fit_size}) ===")
    rng = numpy.random.default_rng(args.seed)
    relatedness, phenotype, _ = simulate(
        args.trace_fit_size,
        args.variants,
        args.tau2,
        args.sigma2,
        args.density,
        args.groups,
        rng,
    )
    reference = fit_reml(
        phenotype,
        materialize(relatedness),
        solver="dense",
        trace_method="exact",
        tolerance=1e-8,
        extra_iterations=0,
        seed=args.seed,
    )
    print(
        f"exact dense REML optimum: tau2={reference.variance_components[0]:.6f} "
        f"sigma2={reference.variance_components[1]:.6f} "
        f"h2={reference.heritability:.6f}"
    )
    print(
        f"{'method':12s} {'num_vectors':>11} {'iters':>6} {'solves':>8} "
        f"{'matvecs':>9} {'seconds':>8} {'h2':>8} {'|dh2|':>8}"
    )
    for method, num_vectors in (
        ("xtrace", budget // 2),
        ("hutchpp", budget),
        ("hutchinson", budget),
    ):
        iterations, matvecs, solves, seconds, estimates = [], [], [], [], []
        for replicate in range(args.trace_fit_repeats):
            started = time.perf_counter()
            fitted = fit_reml(
                phenotype,
                relatedness,
                trace_method=method,
                num_trace_vectors=num_vectors,
                preconditioner_rank=args.preconditioner_rank,
                cg_tol=args.cg_tol,
                seed=replicate,
            )
            seconds.append(time.perf_counter() - started)
            iterations.append(fitted.num_iterations)
            matvecs.append(fitted.num_matvecs)
            solves.append(fitted.num_solves)
            estimates.append(fitted.heritability)
        print(
            f"{method:12s} {num_vectors:>11} {numpy.mean(iterations):>6.1f} "
            f"{numpy.mean(solves):>8.0f} {numpy.mean(matvecs):>9.0f} "
            f"{numpy.mean(seconds):>8.1f} {numpy.mean(estimates):>8.4f} "
            f"{numpy.mean(numpy.abs(numpy.array(estimates) - reference.heritability)):>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    parser.add_argument("--variants", type=int, default=5000)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--groups", type=int, default=25)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--trace-vectors", type=int, default=30)
    parser.add_argument("--preconditioner-rank", type=int, default=100)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--accuracy-size", type=int, default=400)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-budget", type=int, default=60)
    parser.add_argument("--trace-repeats", type=int, default=200)
    parser.add_argument("--trace-fit-size", type=int, default=1000)
    parser.add_argument("--trace-fit-repeats", type=int, default=3)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--trace-compare",
        action="store_true",
        help="compare the trace estimators (and skip the other sections)",
    )
    args = parser.parse_args()

    if args.trace_compare:
        trace_benchmark(args)
        return
    if not args.skip_runtime:
        runtime_benchmark(args)
    if not args.skip_accuracy:
        accuracy_benchmark(args)


if __name__ == "__main__":
    main()
