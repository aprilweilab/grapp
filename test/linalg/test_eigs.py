from contextlib import ExitStack
from grapp.linalg import (
    MatrixSelection,
    PCs,
    eigs as grg_eigs,
    get_pcs_propca,
    sort_by_eigvalues,
)
from parameterized import parameterized
import pygrgl
import numpy
import os
import unittest
import pytest
from scipy.sparse.linalg import eigs as scipy_eigs
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, grg2X, standardize_X, WRAP_GRG_PARAMS

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.pca.grg")
        # Up edges needed for grg2X
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_eigvals(self, backend, wrap_grg):
        stack = self.enterContext(ExitStack())
        X_stand = standardize_X(grg2X(self.grg, diploid=True))

        D = X_stand.T @ X_stand
        evals, evects = scipy_eigs(D, k=15, which="LR")

        # grg_eigs does this for us, because scipy _does not guarantee_ that these are in the
        # correct order.
        sort_by_eigvalues(evals, evects)

        grg_evals, grg_evects = grg_eigs(MatrixSelection.XTX, wrap_grg(self.grg, stack), 15)
        numpy.testing.assert_array_almost_equal(
            numpy.flip(numpy.sort(grg_evals)), grg_evals, 10
        )
        numpy.testing.assert_array_almost_equal(evals, grg_evals, 3)
        # Relative operator residual: the same scale-independent criterion on
        # GRGL, reference, MKL, CUDA and ROCm, outside the eigensolver.
        applied = D @ grg_evects
        residual = numpy.linalg.norm(applied - grg_evects * grg_evals, axis=0)
        scale = numpy.maximum(numpy.linalg.norm(applied, axis=0),
                              numpy.abs(grg_evals) * numpy.linalg.norm(grg_evects, axis=0))
        normalized = numpy.divide(residual, scale, out=residual.copy(), where=scale != 0)
        self.assertTrue(numpy.isfinite(normalized).all())
        self.assertTrue(numpy.all(normalized <= 1e-5), normalized)
        for ev, gev in zip(evects.T, grg_evects.T):
            # Vectors may differ by sign.
            self.assertTrue(numpy.allclose(ev, gev) or numpy.allclose(-ev, gev))

    # Just make sure PCA succeeds
    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_pca_smoketest(self, backend, wrap_grg):
        stack = self.enterContext(ExitStack())
        # Not all recent versions of scipy support passing in a random number generator,
        # so we need to do this instead for testing purposes
        numpy.random.seed(42)
        scores = PCs(wrap_grg(self.grg, stack), k=20).to_numpy()
        pca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "pca.expected.txt"))
        numpy.testing.assert_allclose(numpy.abs(scores), numpy.abs(pca_expect))

    # Just make sure PCA succeeds
    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_propca_smoketest(self, backend, wrap_grg):
        stack = self.enterContext(ExitStack())
        scores, _, _ = get_pcs_propca(wrap_grg(self.grg, stack), k=20, convergence_lim=1e-5)
        propca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "propca.expected.txt"))
        numpy.testing.assert_allclose(
            numpy.abs(scores), numpy.abs(propca_expect), atol=0.005
        )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)


@pytest.mark.parametrize("backend,native,device", [
    pytest.param("reference", False, 0, marks=pytest.mark.reference),
    pytest.param("mkl", False, 0, marks=pytest.mark.mkl),
    *[pytest.param(b, native, 0, marks=getattr(pytest.mark, b))
      for b in ("cusparse", "rocsparse") for native in (False, True)],
    pytest.param("cusparse", True, 1, marks=pytest.mark.cusparse, id="cusparse-native-device1"),
])
@pytest.mark.parametrize("explicit_start", [False, True])
@pytest.mark.parametrize("include_eig", [False, True])
@pytest.mark.parametrize("n_grgs", [1, 2])
def test_pca_optional_start_and_host_outputs(synthetic_grg, tmp_path, backend, native, device, explicit_start, include_eig, n_grgs):
    from grapp.grg_calculator import GRGSpMVCalculator
    from pygrgl_spmv import simple_convert, testing, make_runconfig_pca
    from scipy.linalg import subspace_angles

    if device:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("needs at least two GPU devices")
        assert cupy.cuda.runtime.getDevice() == 0
    artifact = simple_convert(synthetic_grg, tmp_path / "synthetic.grg_spmv")
    raw = pygrgl.load_immutable_grg(str(synthetic_grg), load_up_edges=False)
    size = raw.num_mutations * n_grgs if include_eig else raw.num_individuals
    start = numpy.random.default_rng(0).normal(size=size) if explicit_start else None
    kwargs = dict(k=10, include_eig=include_eig, threads=1, init_vector=start, tol=1e-7)
    reference = PCs([raw] * n_grgs, **kwargs)
    gpu_options = dict(capture=True, native=native, device=device) if backend in {"cusparse", "rocsparse"} else {}
    with testing.load_many([artifact] * n_grgs, backend=backend,
                           req=make_runconfig_pca(maxk=10), **gpu_options) as bound:
        actual = PCs([GRGSpMVCalculator(g) for g in bound], **kwargs)
        if device:
            assert cupy.cuda.runtime.getDevice() == 0
    pcs, ref_pcs = (actual[0], reference[0]) if include_eig else (actual, reference)
    assert pcs.index.name == ref_pcs.index.name == "Individual"
    assert numpy.isfinite(pcs.to_numpy()).all()
    assert numpy.max(subspace_angles(pcs, ref_pcs)) < 1e-6
    if include_eig:
        _, values, vectors = actual
        assert isinstance(values, numpy.ndarray) and isinstance(vectors, numpy.ndarray)
        assert numpy.isfinite(values).all() and numpy.isfinite(vectors).all()
        numpy.testing.assert_allclose(values, reference[1], atol=1e-6, rtol=1e-5)
