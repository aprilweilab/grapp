"""Shared backend selection; GRGL and backend-independent checks always run."""
import pytest

BACKENDS = ("reference", "mkl", "cusparse", "rocsparse")


def pytest_addoption(parser):
    parser.addoption("--backend", default="all", choices=["all", *BACKENDS])


def pytest_configure(config):
    for name in (*BACKENDS, "human_data"):
        config.addinivalue_line("markers", f"{name}: {name} coverage")


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("--backend")
    try:
        from pygrgl_spmv import testing
    except ImportError:
        available = dict.fromkeys(BACKENDS, False)
    else:
        available = {name: testing.is_available(name) for name in BACKENDS
                     if selected in {"all", name}}
    if selected != "all" and not available[selected]:
        raise pytest.UsageError(f"--backend={selected} explicitly requested but {selected} is unavailable")
    for item in items:
        for name in BACKENDS:
            # parameterized.expand gives the existing unittest cases a backend suffix.
            if item.name.endswith("_" + name):
                item.add_marker(getattr(pytest.mark, name))
            if item.get_closest_marker(name) is None:
                continue
            if selected not in {"all", name}:
                item.add_marker(pytest.mark.skip(reason=f"{name} excluded by --backend={selected}"))
            elif not available[name]:
                item.add_marker(pytest.mark.skip(reason=f"{name} unavailable"))


@pytest.fixture
def synthetic_grg(tmp_path):
    import numpy
    import pygrgl
    rng = numpy.random.default_rng(2026)
    matrix = rng.binomial(1, rng.uniform(0.1, 0.8, 150), size=(128, 150))
    g = pygrgl.MutableGRG(128, 2)
    for j in range(matrix.shape[1]):
        node = g.make_node()
        for sample in numpy.flatnonzero(matrix[:, j]):
            g.connect(node, int(sample))
        g.set_num_individual_coals(node, int(numpy.count_nonzero(matrix[::2, j] + matrix[1::2, j] == 2)))
        g.add_mutation(pygrgl.Mutation(j + 1, "C", "A"), node)
    path = tmp_path / "synthetic.grg"
    pygrgl.save_grg(g, str(path))
    return path
