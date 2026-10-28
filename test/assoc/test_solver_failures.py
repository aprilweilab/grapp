"""Failure reporting preserves successful CG arithmetic and bounded secant fitting."""

import numpy as np
import pytest

from grapp.assoc.bolt_inf_core import bolt_conj_grad_solve, CgStats, SolverConvergenceError


@pytest.mark.parametrize("backend", [pytest.param(b, marks=getattr(pytest.mark, b))
                                    for b in ("reference", "mkl", "cusparse", "rocsparse")])
def test_cg_success_and_failure(backend):
    if backend in {"cusparse", "rocsparse"}:
        import cupy as xp
    else:
        xp = np
    diagonal = xp.asarray([1., 2., 3., 4.])
    rhs = xp.asarray([1., 1., 1., 1.])

    def multiply(x, out):
        out[:] = diagonal * x

    kwargs = dict(project=lambda x: x, rel_tol=1e-10)
    stats = CgStats()
    result = bolt_conj_grad_solve([multiply], [rhs], max_iter=4, stats=stats, **kwargs)[0]
    host = xp.asnumpy(result) if xp is not np else result
    np.testing.assert_allclose(host, 1 / np.arange(1, 5))
    with pytest.raises(SolverConvergenceError, match="did not converge"):
        bolt_conj_grad_solve([multiply], [rhs], max_iter=1, **kwargs)

    def nonfinite(x, out):
        out[:] = xp.nan

    with pytest.raises(SolverConvergenceError, match="non-finite"):
        bolt_conj_grad_solve([nonfinite], [rhs], max_iter=4, **kwargs)
    for rel_tol, max_iter in ((0., 4), (np.inf, 4), (1e-10, 0)):
        with pytest.raises(ValueError, match="CG requires"):
            bolt_conj_grad_solve([multiply], [rhs], max_iter=max_iter,
                                rel_tol=rel_tol, project=lambda x: x)


@pytest.mark.parametrize("outcome", ["limit", "converged", "nonfinite"])
def test_secant_limit_returns_best_finite_with_warning(monkeypatch, caplog, outcome):
    from types import SimpleNamespace
    from grapp.assoc import bolt_inf_core as core

    monkeypatch.setattr(core, "_generate_bolt_mc_components", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(core, "log_delta_from_h2", lambda ops, h2: h2)
    monkeypatch.setattr(core, "h2_from_log_delta", lambda ops, delta: delta)
    estimates = []

    def evaluate(*args, log_delta, **kwargs):
        i = len(estimates)
        estimate = SimpleNamespace(log_delta=log_delta, f_reml=(-1.)**(i+1), sigma2_k=1.,
                                   all_hinv_y=np.asarray([i]))
        if outcome == "converged":
            estimate.f_reml = log_delta - 0.125
        if outcome == "nonfinite" and i == 2:
            estimate.f_reml = np.nan
        estimates.append(estimate)
        return estimate

    monkeypatch.setattr(core, "_compute_mc_scaling", evaluate)
    kwargs = dict(mc_trials=3, seed=0, rel_tol=5e-4, max_iter=10000, stats=CgStats())
    if outcome == "nonfinite":
        with pytest.raises(SolverConvergenceError, match="Secant.*non-finite"):
            core.fit_bolt_variance_components(SimpleNamespace(n=10), None, **kwargs)
    else:
        fit = core.fit_bolt_variance_components(SimpleNamespace(n=10), None, **kwargs)
        assert len(estimates) == (7 if outcome == "limit" else 3)
        assert fit.log_delta == estimates[2].log_delta
        assert fit.all_hinv_y is estimates[2].all_hinv_y
        assert fit.secant_converged == (outcome == "converged")
        assert fit.secant_steps == len(estimates) - 2
        assert fit.cg_stats is kwargs["stats"]
        assert np.isfinite([fit.log_delta, fit.sigma_g2, fit.sigma_e2, fit.h2]).all()
        assert ("Secant iteration for h2 estimation may not have converged" in caplog.text) == (outcome == "limit")
