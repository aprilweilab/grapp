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
| `trace(P V_i)`          | XTrace (Epperly et al. 2024), variance `O(1/m^2)`          |
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
| `trace_method`         | `xtrace`  | `hutchinson` is half the cost per vector and much noisier             |
| `cg_tol`               | `1e-5`    | relative residual of each solve                                       |
| `preconditioner_rank`  | 100       | sketched once per component; raise it when a few eigenvalues dominate |
| `tolerance`            | 0.05      | relative change that counts as converged                              |
| `extra_iterations`     | 15        | averaging window after convergence                                    |
| `solver`               | `cg`      | `dense` materializes `V`; exact, only for small `N`                   |

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
* Frangella Z, Tropp JA, Udell M (2021). Randomized Nystrom preconditioning.
  *SIMAX* 44(2):718-752.
* Wu Y, Sankararaman S (2018). A scalable estimator of SNP heritability for
  biobank-scale data. *Bioinformatics* 34(13):i187-i194.
* Patterson HD, Thompson R (1971). Recovery of inter-block information when
  block sizes are unequal. *Biometrika* 58(3):545-554.
