"""
One AI-REML solve per process, for wall-clock and peak-RAM time trials.

The other benchmark script (``aireml_benchmark.py``) runs many fits inside one
process, which is convenient but useless for measuring peak memory: the high
-water mark of the whole process is whatever the largest fit needed, plus
everything the earlier fits left behind.  This script does exactly one solve
and exits, so an external profiler sees the cost of that solve alone::

    /usr/bin/time -l python examples/aireml_timing.py solve --dataset data/n16000
    /usr/bin/time -v python examples/aireml_timing.py solve --dataset data/n16000   # GNU

It also reports its own peak RSS, so the numbers are available even without a
profiler.

Two modes:

``make``
    Write a dataset to disk: a sparse genotype matrix and a phenotype drawn
    from the LMM.  Done once per size, so the timed run does not pay for
    simulation.

``solve``
    Load a dataset and fit it.  The relatedness operator comes either from the
    stored genotype matrix (``--dataset``) or, when ``grapp`` and ``pygrgl``
    are installed, from a GRG (``--grg``), which is the apples-to-apples
    comparison against ``grapp pca``.

``trials``
    Drive a sweep: make any missing datasets, run each solve in its own
    process under an external profiler, parse the peak RSS back out, and print
    one table.  Pass ``--grg`` to add a ``grapp pca`` row as a memory
    baseline on the same data.
"""

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy
import scipy.sparse

from aireml import fit_reml, grm_from_genotypes

#: Bumped whenever make_dataset changes what it writes, so that a cached
#: dataset from an older version is regenerated instead of being compared
#: against freshly generated ones.
GENERATOR_VERSION = 2


def peak_rss_mb() -> float:
    """
    Peak resident set size of this process, in MiB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux, which is a classic
    way to be wrong by 1024x.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def make_dataset(args):
    """Simulate and store one dataset."""
    rng = numpy.random.default_rng(args.seed)
    started = time.perf_counter()

    # Background variation.  Positions are sampled with replacement, which is
    # O(nnz); scipy.sparse.random instead samples without replacement from the
    # whole size-by-variants grid, which takes minutes rather than a fraction
    # of a second once N reaches 64000.  About 0.5% of positions collide and
    # simply sum, which is harmless for synthetic genotypes.
    target_nnz = int(round(args.size * args.variants * args.density))
    genotypes = scipy.sparse.coo_matrix(
        (
            rng.integers(1, 3, size=target_nnz).astype(numpy.float64),
            (
                rng.integers(0, args.size, size=target_nnz),
                rng.integers(0, args.variants, size=target_nnz),
            ),
        ),
        shape=(args.size, args.variants),
    ).tocsr()

    # Group structure, so the GRM has a few large eigenvalues like a real
    # sample with relatedness rather than a near-diagonal one.
    membership = rng.integers(0, args.groups, size=args.size)
    block = max(1, args.variants // (4 * args.groups))
    row_parts, col_parts = [], []
    for group in range(args.groups):
        members = numpy.flatnonzero(membership == group)
        if members.size == 0:
            continue
        start = (group * block) % max(1, args.variants - block)
        columns = numpy.arange(start, start + block)
        row_parts.append(numpy.tile(members, block))
        col_parts.append(numpy.repeat(columns, members.size))
    rows = numpy.concatenate(row_parts)
    cols = numpy.concatenate(col_parts)
    shared = scipy.sparse.coo_matrix(
        (numpy.ones(rows.size), (rows, cols)), shape=(args.size, args.variants)
    ).tocsr()
    genotypes = (genotypes + shared).tocsr()

    # Normalize so the GRM has mean diagonal 1 and tau2 is a genetic variance.
    scale = float(genotypes.multiply(genotypes).sum()) / args.size
    effects = rng.standard_normal(args.variants)
    genetic = genotypes @ effects / numpy.sqrt(scale)
    phenotype = (
        1.5
        + numpy.sqrt(args.tau2) * genetic
        + numpy.sqrt(args.sigma2) * rng.standard_normal(args.size)
    )

    os.makedirs(os.path.dirname(args.dataset) or ".", exist_ok=True)
    scipy.sparse.save_npz(args.dataset + ".genotypes.npz", genotypes)
    numpy.save(args.dataset + ".phenotype.npy", phenotype)
    with open(args.dataset + ".meta.json", "w") as handle:
        json.dump(
            {
                "generator": GENERATOR_VERSION,
                "size": args.size,
                "variants": args.variants,
                "nnz": int(genotypes.nnz),
                "scale": scale,
                "tau2": args.tau2,
                "sigma2": args.sigma2,
                "groups": args.groups,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    megabytes = sum(
        os.path.getsize(args.dataset + suffix)
        for suffix in (".genotypes.npz", ".phenotype.npy")
    ) / (1024.0 * 1024.0)
    print(
        f"wrote {args.dataset}.* : N={args.size} M={args.variants} "
        f"nnz={genotypes.nnz} on-disk={megabytes:.1f} MiB "
        f"({time.perf_counter() - started:.1f}s)"
    )


def _load_from_dataset(args):
    genotypes = scipy.sparse.load_npz(args.dataset + ".genotypes.npz")
    phenotype = numpy.load(args.dataset + ".phenotype.npy")
    with open(args.dataset + ".meta.json") as handle:
        meta = json.load(handle)
    operator = grm_from_genotypes(genotypes, scale=meta["scale"])
    return (
        operator,
        phenotype,
        meta,
        f"{genotypes.shape[0]}x{genotypes.shape[1]} sparse",
    )


def _load_from_grg(args):
    """
    Relatedness straight off a GRG, for comparison against ``grapp pca``.

    This is the only part of the timing harness that touches grapp; aireml
    itself never imports it.
    """
    try:
        from grapp.grg_calculator import load_grg_calculator
        from grapp.linalg.ops_scipy import SciPyStdXOperator
    except ImportError as error:  # pragma: no cover - optional path
        raise SystemExit(
            f"--grg needs grapp's dependencies (pygrgl, pandas, ...): {error}\n"
            "Install them with:  pip install -r requirements.txt"
        )
    calculator = load_grg_calculator(args.grg)
    genotypes = SciPyStdXOperator(calculator)
    num_samples = genotypes.shape[0]
    if args.phenotype:
        phenotype = numpy.load(args.phenotype)
    else:
        rng = numpy.random.default_rng(args.seed)
        effects = rng.standard_normal(genotypes.shape[1])
        genetic = numpy.asarray(genotypes.matvec(effects))
        genetic /= numpy.std(genetic) or 1.0
        phenotype = (
            1.5
            + numpy.sqrt(args.tau2) * genetic
            + numpy.sqrt(args.sigma2) * rng.standard_normal(num_samples)
        )
    operator = grm_from_genotypes(genotypes)
    return operator, phenotype, {"size": num_samples}, f"GRG {args.grg}"


def solve(args):
    load_started = time.perf_counter()
    if args.grg:
        operator, phenotype, meta, source = _load_from_grg(args)
    else:
        operator, phenotype, meta, source = _load_from_dataset(args)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = peak_rss_mb()

    started = time.perf_counter()
    result = fit_reml(
        phenotype,
        operator,
        trace_method=args.trace_method,
        num_trace_vectors=args.trace_vectors,
        preconditioner_rank=args.preconditioner_rank,
        cg_tol=args.cg_tol,
        extra_iterations=args.extra_iterations,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - started

    row = {
        "source": source,
        "N": meta.get("size"),
        "load_s": round(load_seconds, 2),
        "fit_s": round(fit_seconds, 2),
        "iters": result.num_iterations,
        "solves": result.num_solves,
        "matvecs": result.num_matvecs,
        "tau2": round(float(result.variance_components[0]), 6),
        "sigma2": round(float(result.variance_components[-1]), 6),
        "h2": round(result.heritability, 6),
        "converged": result.converged,
        "rss_after_load_mb": round(rss_after_load, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    if args.json:
        print(json.dumps(row))
    else:
        print(
            f"N={row['N']} source={source}\n"
            f"  load   {row['load_s']:.2f}s   fit {row['fit_s']:.2f}s   "
            f"iters {row['iters']}   solves {row['solves']}   "
            f"matvecs {row['matvecs']}\n"
            f"  tau2 {row['tau2']:.6f}  sigma2 {row['sigma2']:.6f}  "
            f"h2 {row['h2']:.6f}  converged {row['converged']}\n"
            f"  peak RSS {row['peak_rss_mb']:.1f} MiB "
            f"(after load, before fit: {row['rss_after_load_mb']:.1f} MiB)"
        )


# --- external profiler -------------------------------------------------------
#
# GNU time reports "Maximum resident set size (kbytes): N" under -v; BSD/macOS
# time reports "N  maximum resident set size" in *bytes* under -l.  Both are
# parsed so the same sweep works on a laptop and on a cluster node.

_GNU_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_BSD_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)
_GNU_WALL = re.compile(r"Elapsed \(wall clock\) time [^:]*:\s*([0-9:.]+)")
_BSD_WALL = re.compile(r"^\s*([0-9.]+)\s+real", re.MULTILINE)


def find_profiler():
    """
    Locate a resource-reporting ``time`` and return ``(argv_prefix, flavour)``.

    Prefers GNU time, which reports more and is what ``-v`` refers to.  On
    macOS that is ``gtime`` from ``brew install gnu-time``; the built-in
    ``/usr/bin/time`` only understands ``-l``.
    """
    for name in ("gtime", "gnutime"):
        path = shutil.which(name)
        if path:
            return [path, "-v"], "gnu"
    if sys.platform != "darwin" and shutil.which("/usr/bin/time"):
        probe = subprocess.run(
            ["/usr/bin/time", "-v", "true"], capture_output=True, text=True
        )
        if "Maximum resident set size" in probe.stderr:
            return ["/usr/bin/time", "-v"], "gnu"
    if os.path.exists("/usr/bin/time"):
        return ["/usr/bin/time", "-l"], "bsd"
    return None, None


def parse_profiler(text, flavour):
    """Return ``(peak_rss_mb, wall_seconds)`` from a profiler's stderr."""
    rss_mb = wall = None
    if flavour == "gnu":
        match = _GNU_RSS.search(text)
        if match:
            rss_mb = int(match.group(1)) / 1024.0
        match = _GNU_WALL.search(text)
        if match:
            parts = [float(p) for p in match.group(1).split(":")]
            wall = parts[-1]
            for index, value in enumerate(reversed(parts[:-1]), start=1):
                wall += value * (60**index)
    else:
        match = _BSD_RSS.search(text)
        if match:
            rss_mb = int(match.group(1)) / (1024.0 * 1024.0)
        match = _BSD_WALL.search(text)
        if match:
            wall = float(match.group(1))
    return rss_mb, wall


def _error_tail(text):
    """
    Last meaningful line of a failed child's stderr.

    The profiler appends its own resource table to stderr, so the literal last
    line is usually "peak memory footprint" rather than the error.
    """
    noise = (
        "maximum resident set size",
        "average shared memory",
        "average unshared",
        "page reclaims",
        "page faults",
        "swaps",
        "block input",
        "block output",
        "messages sent",
        "messages received",
        "signals received",
        "voluntary",
        "involuntary",
        "instructions retired",
        "cycles elapsed",
        "peak memory footprint",
        "real",
        "Command being timed",
        "User time",
        "System time",
        "Percent of CPU",
        "Elapsed (wall clock)",
        "Average ",
        "Major (requiring",
        "Minor (reclaiming",
        "Swaps",
        "File system",
        "Socket",
        "Signals",
        "Exit status",
        "Maximum resident",
    )
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if not any(marker in line for marker in noise):
            return line
    return "(no error line found)"


def trials(args):
    profiler, flavour = find_profiler()
    if profiler is None:
        print("no external 'time' found; falling back to self-reported RSS only")
    else:
        print(f"profiler: {' '.join(profiler)} ({flavour})")
    print(f"python  : {sys.executable}")
    print(
        f"settings: trace={args.trace_method} vectors={args.trace_vectors} "
        f"precond-rank={args.preconditioner_rank} cg-tol={args.cg_tol} "
        f"extra-iters={args.extra_iterations}"
    )

    header = (
        f"{'N':>8} {'nnz':>10} {'load_s':>7} {'fit_s':>8} {'iters':>6} "
        f"{'matvecs':>9} {'h2':>7} {'RSS_MiB':>8} {'s/1e3':>7} {'KiB/ind':>8}"
    )
    rows = []
    print("\n=== AI-REML ===")
    print(header)
    for size in args.sizes:
        prefix = os.path.join(args.data_dir, f"n{size}")
        stale = True
        if os.path.exists(prefix + ".genotypes.npz") and os.path.exists(
            prefix + ".meta.json"
        ):
            with open(prefix + ".meta.json") as handle:
                cached = json.load(handle)
            stale = (
                cached.get("generator") != GENERATOR_VERSION
                or cached.get("variants") != args.variants
                or cached.get("density") != args.density
                or cached.get("groups") != args.groups
            )
        if stale:
            subprocess.check_call(
                [
                    sys.executable,
                    __file__,
                    "make",
                    "--dataset",
                    prefix,
                    "--size",
                    str(size),
                    "--variants",
                    str(args.variants),
                    "--density",
                    str(args.density),
                    "--groups",
                    str(args.groups),
                    "--seed",
                    str(args.seed),
                ],
                stdout=subprocess.DEVNULL,
            )
        command = [
            sys.executable,
            __file__,
            "solve",
            "--dataset",
            prefix,
            "--json",
            "--trace-method",
            args.trace_method,
            "--trace-vectors",
            str(args.trace_vectors),
            "--preconditioner-rank",
            str(args.preconditioner_rank),
            "--cg-tol",
            str(args.cg_tol),
            "--extra-iterations",
            str(args.extra_iterations),
            "--seed",
            str(args.seed),
        ]
        if profiler:
            command = profiler + command
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(f"{size:>8}  FAILED: {_error_tail(done.stderr)}")
            continue
        row = json.loads(done.stdout.strip().splitlines()[-1])
        rss, _ = parse_profiler(done.stderr, flavour) if profiler else (None, None)
        rss = rss if rss is not None else row["peak_rss_mb"]
        with open(prefix + ".meta.json") as handle:
            nnz = json.load(handle)["nnz"]
        print(
            f"{row['N']:>8} {nnz:>10} {row['load_s']:>7.2f} {row['fit_s']:>8.2f} "
            f"{row['iters']:>6} {row['matvecs']:>9} {row['h2']:>7.4f} "
            f"{rss:>8.1f} {1e3 * row['fit_s'] / row['N']:>7.3f} "
            f"{1024.0 * rss / row['N']:>8.2f}"
        )
        rows.append((row["N"], row["fit_s"], rss))

    if len(rows) > 1:
        sizes = numpy.array([r[0] for r in rows], dtype=float)
        times = numpy.array([r[1] for r in rows], dtype=float)
        memory = numpy.array([r[2] for r in rows], dtype=float)
        print(
            f"\nempirical scaling: time ~ N^{numpy.polyfit(numpy.log(sizes), numpy.log(times), 1)[0]:.2f}"
            f"   peak RSS ~ N^{numpy.polyfit(numpy.log(sizes), numpy.log(memory), 1)[0]:.2f}"
        )

    if args.grg:
        print("\n=== same GRG: grapp pca vs AI-REML (peak RSS) ===")
        print(f"{'what':28} {'wall_s':>8} {'RSS_MiB':>9}")
        jobs = [
            (
                "grapp pca -d %d" % args.pca_dimensions,
                ["grapp", "pca", "-d", str(args.pca_dimensions), args.grg],
            ),
            (
                "aireml solve (GRG-backed)",
                [
                    sys.executable,
                    __file__,
                    "solve",
                    "--grg",
                    args.grg,
                    "--json",
                    "--trace-method",
                    args.trace_method,
                    "--trace-vectors",
                    str(args.trace_vectors),
                    "--preconditioner-rank",
                    str(args.preconditioner_rank),
                    "--seed",
                    str(args.seed),
                ],
            ),
        ]
        for label, command in jobs:
            if shutil.which(command[0]) is None and command[0] != sys.executable:
                print(f"{label:28} {'--':>8} {'--':>9}   ({command[0]} not on PATH)")
                continue
            full = (profiler + command) if profiler else command
            started = time.perf_counter()
            done = subprocess.run(full, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            if done.returncode != 0:
                print(
                    f"{label:28} {'FAILED':>8} {'--':>9}   "
                    f"{_error_tail(done.stderr)[:90]}"
                )
                continue
            rss, wall = (
                parse_profiler(done.stderr, flavour) if profiler else (None, None)
            )
            print(
                f"{label:28} {(wall or elapsed):>8.2f} "
                f"{(rss if rss is not None else float('nan')):>9.1f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--tau2", type=float, default=1.0)
    common.add_argument("--sigma2", type=float, default=1.0)

    maker = sub.add_parser("make", parents=[common], help="write a dataset")
    maker.add_argument("--dataset", required=True, help="path prefix to write")
    maker.add_argument("--size", type=int, required=True, help="number of individuals")
    maker.add_argument("--variants", type=int, default=5000)
    maker.add_argument("--density", type=float, default=0.01)
    maker.add_argument("--groups", type=int, default=25)
    maker.set_defaults(func=make_dataset)

    solver = sub.add_parser("solve", parents=[common], help="run one fit")
    source = solver.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="path prefix written by 'make'")
    source.add_argument("--grg", help="a GRG file (needs grapp's dependencies)")
    solver.add_argument("--phenotype", help="optional .npy phenotype, for --grg")
    solver.add_argument("--trace-method", default="xtrace")
    solver.add_argument("--trace-vectors", type=int, default=30)
    solver.add_argument("--preconditioner-rank", type=int, default=100)
    solver.add_argument("--cg-tol", type=float, default=1e-5)
    solver.add_argument("--extra-iterations", type=int, default=15)
    solver.add_argument("--max-iterations", type=int, default=100)
    solver.add_argument("--json", action="store_true", help="one JSON line")
    solver.set_defaults(func=solve)

    sweep = sub.add_parser("trials", parents=[common], help="drive a size sweep")
    sweep.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 2000, 4000, 8000, 16000, 32000],
    )
    sweep.add_argument("--data-dir", default="timing-data")
    sweep.add_argument("--variants", type=int, default=5000)
    sweep.add_argument("--density", type=float, default=0.01)
    sweep.add_argument("--groups", type=int, default=25)
    sweep.add_argument("--trace-method", default="xtrace")
    sweep.add_argument("--trace-vectors", type=int, default=30)
    sweep.add_argument("--preconditioner-rank", type=int, default=100)
    sweep.add_argument("--cg-tol", type=float, default=1e-5)
    sweep.add_argument("--extra-iterations", type=int, default=15)
    sweep.add_argument("--grg", help="GRG file for the grapp-pca memory baseline")
    sweep.add_argument("--pca-dimensions", type=int, default=10)
    sweep.set_defaults(func=trials)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
