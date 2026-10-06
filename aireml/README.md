# aireml

Average-information REML (AI-REML) for variance components whose covariance
structure is only available as a **linear operator**.

This is a small, self-contained implementation of the matrix-free AI-REML
algorithm described in

> Lee H, Pope NS, Kelleher J, Gorjanc G, Ralph PL (2026).
> *Genetic prediction with ARG-powered linear algebra.*
> GENETICS 233(1), iyag074. https://doi.org/10.1093/genetics/iyag074

which in turn builds on Gilmour, Thompson and Cullis (1995).

It depends only on `numpy` and `scipy`, and knows nothing about ARGs, GRGs,
tree sequences, or genotype file formats. You give it a phenotype vector `y`
and one or more `scipy.sparse.linalg.LinearOperator`s, and it estimates the
variance components.

## Why an operator?

For `N` individuals, a genetic relatedness matrix costs `O(N^2)` to store and
`O(N^3)` to factorize, which is hopeless at biobank scale. But every quantity
AI-REML needs can be written in terms of products `B @ x`, and many
representations of relatedness can do that product in nearly linear time: an
ARG/tree sequence (`tskit.TreeSequence.genetic_relatedness_vector`), a GRG, a
sparse or memory-mapped genotype matrix, a low-rank factor, a pedigree.

So `aireml` asks only for the product and supplies the rest:

| need                    | how                                                        |
| ----------------------- | ---------------------------------------------------------- |
| `V^-1 b`                | preconditioned conjugate gradients                         |
| preconditioner          | randomized Nystrom (Frangella et al. 2021), sketched once  |
| `trace(P V_i)`          | XTrace or Hutch++, variance `O(1/m^2)`                     |
| starting values         | randomized Haseman-Elston (Wu and Sankararaman 2018)       |
| curvature               | average information -- exact, no trace estimate needed     |

## Install

```
pip install numpy scipy
```

then either `pip install -e .` from a checkout of this directory's parent, or
simply copy the `aireml/` directory into your project: it is pure Python with
no other dependencies.

## Use

```python
import numpy
from aireml import fit_reml, grm_from_genotypes

genotypes = ...                       # anything with a matvec: (N, M)
grm = grm_from_genotypes(genotypes)   # N x N LinearOperator, Z Z^T / M

result = fit_reml(y, grm, covariates=covariates)
print(result.summary())

result.variance_components   # array([tau2, sigma2])
result.heritability          # tau2 / (tau2 + sigma2)
result.heritability_stderr   # delta method, from the average information
result.fixed_effects         # GLS estimate of b
result.blup()                # tau2 * B P y, genetic values of the fitted set
```

Bring your own operator -- nothing above is specific to genotypes:

```python
from scipy.sparse.linalg import LinearOperator

relatedness = LinearOperator(
    shape=(n, n), matvec=my_relatedness_product, dtype=float
)
result = fit_reml(y, relatedness)
```

Predict genetic values for individuals who were not fitted, given the
cross-covariance block between them and the fitted individuals (Lee et al.
2026, equation 22):

```python
predicted = result.predict(cross_covariance_operator)   # (N_new,)
```

Several variance components work the same way; the residual `sigma^2 I` is
always appended automatically:

```python
result = fit_reml(y, [grm_chr1, grm_chr2, grm_chr3])
result.variance_components   # array([tau2_1, tau2_2, tau2_3, sigma2])
```

## The algorithm

With `V = sum_i theta_i V_i` and `P` the REML projection matrix,

```
gradient_i = trace(P V_i) - y^T P V_i P y
AI_ij      = y^T P V_i P V_j P y
theta      <- theta - AI^-1 gradient
```

The average information is the mean of the observed Hessian and the Fisher
information of the REML objective. The expensive trace terms that appear in
each of them cancel in the average, so the curvature is computed *exactly*
from a handful of solves, and only the gradient is stochastic. That is what
makes the optimization stable despite the randomized linear algebra.

Following the paper, iteration stops once the estimates move by less than
`tolerance` (default 5%), runs `extra_iterations` more (default 15), and
returns the average of that window, which averages out the gradient noise.
Set `extra_iterations=0` with `trace_method="exact"` to get a deterministic
fit that converges to machine precision -- that is the mode the tests use to
check the updates against a brute-force maximization of the likelihood.

## Choosing the knobs

| parameter              | default   | note                                                                 |
| ---------------------- | --------- | -------------------------------------------------------------------- |
| `num_trace_vectors`    | 50        | dominates cost: `2 x this` solves per component per iteration         |
| `trace_method`         | `xtrace`  | or `hutchpp`; `hutchinson` is much noisier. See below                 |
| `cg_tol`               | `1e-5`    | relative residual of each solve                                       |
| `preconditioner_rank`  | 100       | sketched once per component; raise it when a few eigenvalues dominate |
| `tolerance`            | 0.05      | relative change that counts as converged                              |
| `extra_iterations`     | 15        | averaging window after convergence                                    |
| `solver`               | `cg`      | `dense` materializes `V`; exact, only for small `N`                   |

## Trace estimators

`trace(P V_i)` is the only quantity in the gradient that cannot be reduced to
a few products, and each product against `P` is a CG solve -- so this choice
drives the runtime. Three unbiased estimators are available, and the fair way
to compare them is at equal *matrix-vector* count, since XTrace spends two
products per test vector while the other two spend one:

| estimator | products | relative SD at 60 products | notes |
| --------- | -------- | -------------------------- | ----- |
| `hutchinson` | `m` | 1.00 | the classical estimator; variance `O(1/m)` |
| `hutchpp` | `3 * (m // 3)` | 0.069 | Hutch++ (Meyer et al. 2021), Algorithm 1 |
| `xtrace` | `2 * m` | 0.053 | Epperly et al. (2024), leave-one-out |

Measured on a 400x400 symmetric matrix with eigenvalues `k^-1.5` over 300
repeats; reproduce with

```
python examples/aireml_benchmark.py --trace-compare
```

Both sketch-and-correct estimators are an order of magnitude better than
plain Hutchinson, and XTrace has a somewhat smaller constant than Hutch++ for
the same number of products, which is why it stays the default. The ordering
is also asserted in `test_variance_ranking_at_equal_matvec_budget`.

One caveat, measured rather than assumed: **this ranking does not carry
through to a better fit.** At N=1500 over 12 seeds at equal product budget
(~25,800 products each), all three estimators converge every time and land on
the same answer:

| estimator | sec | mean h2 | SD h2 | max err |
| --------- | ---:| -------:| -----:| -------:|
| `xtrace` | 14.1 | 0.49602 | 0.00176 | 0.00433 |
| `hutchpp` | 12.9 | 0.49743 | 0.00221 | 0.00380 |
| `hutchinson` | 12.2 | 0.49640 | 0.00168 | 0.00319 |

(exact REML optimum 0.496796). The spreads are within each other's
uncertainty, so even plain Hutchinson is not measurably worse *here*. Two
structural reasons: the average information is computed exactly, so trace
noise perturbs only the search direction and never the curvature; and the
default stopping rule returns the mean of the last 15 iterates, which
averages the remaining noise across iterations.

The wall-clock ordering is the one real difference, and it tracks dense
overhead rather than solves: XTrace's leave-one-out downdate costs a
triangular solve and several m-by-m products per estimate, which is why it is
the slowest of the three despite spending the same number of solves.

Two practical consequences, both bigger than the estimator choice:

* `extra_iterations` dominates the cost/accuracy trade. Dropping it to 0 on
  the same problem converges in 2 iterations and 1.5s instead of 17 and 13s,
  at a worst-case error of 0.018 rather than 0.004 -- 8x cheaper for 4x the
  error. Worth tuning deliberately.
* `num_trace_vectors` matters more than which estimator spends it.

`xtrace` remains the default: it has the lowest variance per product, it is
what Lee et al. (2026) use, and a ~9% wall-clock gap on a single benchmark is
too thin to justify changing it. Switch to `hutchpp` if you want the speed
back and are satisfied by the table above.

Hutch++ splits its budget three ways -- sketch, exact trace on the sketched
subspace, Hutchinson correction on the complement -- and is the better choice
if you want a simpler estimator or care about the dense `O(N m^2)` overhead
that XTrace's leave-one-out downdate adds on top of the solves. The
implementation follows the same Algorithm 1 as
`pylops.utils.estimators.trace_hutchpp`, and is cross-checked against it in
the test suite (pylops is not a dependency; the check skips if it is absent).

## Cost

Per AI-REML iteration, with `m` genetic components and `t` trace vectors:

* `2 t (m + 1)` solves against `V` for the traces,
* `m + 1` solves for the average information,
* `C + 1` solves for the covariates and the phenotype,

each solve being a CG run of a few dozen products against `V`. Nothing is
`O(N^2)`.

## Accuracy

`test/aireml/` checks, among other things, that

* the analytic gradient matches finite differences of the REML objective,
* the average information matches the explicit `y^T P V_i P V_j P y`,
* AI-REML with exact traces reaches the same optimum as a brute-force
  Nelder-Mead maximization of the restricted likelihood (to `1e-5` relative),
* the matrix-free stochastic path lands within a few percent of that optimum,
* the estimates are unbiased across simulation replicates, and the reported
  standard errors track their empirical spread,
* BLUPs match the closed-form `tau^2 B[new, fitted] P y`.

`examples/aireml_benchmark.py` measures runtime scaling and compares AI-REML
against Haseman-Elston.

## References

* Gilmour AR, Thompson R, Cullis BR (1995). Average information REML.
  *Biometrics* 51(4):1440-1450.
* Lee H, Pope NS, Kelleher J, Gorjanc G, Ralph PL (2026). Genetic prediction
  with ARG-powered linear algebra. *GENETICS* 233(1):iyag074.
* Epperly E, Tropp JA, Webber RJ (2024). XTrace: making the most of every
  sample in stochastic trace estimation. *SIMAX* 45(1):1-23.
* Meyer RA, Musco C, Musco C, Woodruff DP (2021). Hutch++: optimal stochastic
  trace estimation. *SOSA* 2021:142-155.
* Frangella Z, Tropp JA, Udell M (2021). Randomized Nystrom preconditioning.
  *SIMAX* 44(2):718-752.
* Wu Y, Sankararaman S (2018). A scalable estimator of SNP heritability for
  biobank-scale data. *Bioinformatics* 34(13):i187-i194.
* Patterson HD, Thompson R (1971). Recovery of inter-block information when
  block sizes are unequal. *Biometrika* 58(3):545-554.
