from abc import ABC, abstractmethod
import pygrgl
import numpy
import threading
import concurrent.futures
import contextlib
from typing import Optional, Union, Dict, Callable, List, Any

try:
    import pygrgl_spmv
except ImportError:
    pygrgl_spmv = None  # typing: ignore

try:
    import cupy
except ImportError:
    cupy = None  # typing: ignore


def _scipy_operator_table() -> Dict[tuple, Callable]:
    # Lazy import to avoid a circular import: grapp.linalg and grapp.util both import
    # grapp.grg_calculator. Keys are (op, standardized, multi). "X" and "XT" share a class
    # (distinguished by the direction passed to the constructor). The non-standardized multi-XXT
    # operator is absent. "EIGSH" maps to the sparse symmetric eigensolver.
    # All interfaces take numpy arrays and return numpy arrays, conversion to/from cupy arrays
    # should be done internally!
    from grapp.linalg import ops_scipy as m
    from scipy.sparse.linalg import eigsh

    return {
        ("X", False, False): m.SciPyXOperator,
        ("XT", False, False): m.SciPyXOperator,
        ("XTX", False, False): m.SciPyXTXOperator,
        ("XXT", False, False): m.SciPyXXTOperator,
        ("X", True, False): m.SciPyStdXOperator,
        ("XT", True, False): m.SciPyStdXOperator,
        ("XTX", True, False): m.SciPyStdXTXOperator,
        ("XXT", True, False): m.SciPyStdXXTOperator,
        ("X", False, True): m.MultiSciPyXOperator,
        ("XT", False, True): m.MultiSciPyXOperator,
        ("XTX", False, True): m.MultiSciPyXTXOperator,
        ("X", True, True): m.MultiSciPyStdXOperator,
        ("XT", True, True): m.MultiSciPyStdXOperator,
        ("XTX", True, True): m.MultiSciPyStdXTXOperator,
        ("XXT", True, True): m.MultiSciPyStdXXTOperator,
        ("EIGSH", False, False): eigsh,
        ("EIGSH", True, False): eigsh,
    }


def _cupy_operator_table() -> Dict[tuple, Callable]:
    assert (
        cupy is not None
    ), "cupy not installed; try 'pip install cupy' or use a different backend"
    from grapp.linalg import ops_cupy as m
    from cupyx.scipy.sparse.linalg import eigsh

    def _eigsh(A, **kwargs):
        def _convert_if_present(kwargs, key):
            if key in kwargs:
                kwargs[key] = cupy.asarray(kwargs[key])

        _convert_if_present(kwargs, "M")
        _convert_if_present(kwargs, "v0")
        eigval, eigvect = eigsh(A, **kwargs)
        cupy.cuda.Device().synchronize()
        return cupy.asnumpy(eigval), cupy.asnumpy(eigvect)

    return {
        ("X", False, False): m.CuPyXOperator,
        ("XT", False, False): m.CuPyXOperator,
        ("XTX", False, False): m.CuPyXTXOperator,
        ("XXT", False, False): m.CuPyXXTOperator,
        ("X", True, False): m.CuPyStdXOperator,
        ("XT", True, False): m.CuPyStdXOperator,
        ("XTX", True, False): m.CuPyStdXTXOperator,
        ("XXT", True, False): m.CuPyStdXXTOperator,
        ("X", False, True): m.MultiCuPyXOperator,
        ("XT", False, True): m.MultiCuPyXOperator,
        ("XTX", False, True): m.MultiCuPyXTXOperator,
        ("XXT", False, True): m.MultiCuPyXXTOperator,
        ("X", True, True): m.MultiCuPyStdXOperator,
        ("XT", True, True): m.MultiCuPyStdXOperator,
        ("XTX", True, True): m.MultiCuPyStdXTXOperator,
        ("XXT", True, True): m.MultiCuPyStdXXTOperator,
        ("EIGSH", False, False): _eigsh,
        ("EIGSH", True, False): _eigsh,
    }


def _select_operator_cls(
    backend: str, op: str, standardized: bool, multi: bool
) -> Callable:
    """
    Map (op, standardized, multi) to a LinearOperator class (or, for the non-matrix ops, a callable)
    for the given backend, hiding the SciPy/CuPy choice from downstream code. Returns the
    class/callable, not an instance/result; the caller invokes it with whatever arguments it needs
    (grg(s), direction, freqs, ...). The non-matrix ops are: ``"FREQ"`` (allele-frequency function),
    ``"EIGSH"`` (sparse symmetric eigensolver), and ``"TO_NUMPY"``/``"FROM_NUMPY"`` (host<->backend
    array converters).

    :param backend: ``"SciPy"`` or ``"CuPy"``.
    :param op: One of ``"X"``, ``"XT"``, ``"XTX"``, ``"XXT"``, ``"FREQ"``, ``"EIGSH"``,
        ``"TO_NUMPY"``, ``"FROM_NUMPY"`` (case-insensitive).
    :param standardized: Select the standardized (mean/variance-scaled) operator. Ignored for the
        non-matrix ops.
    :param multi: Select the multi-GRG variant.
    :raises ValueError: For an unrecognized ``op`` or ``backend``.
    :raises NotImplementedError: When the backend has no entry for the requested combination
        (e.g. the non-standardized multi-XXT operator does not exist for SciPy, and the non-matrix
        ops have no multi variant).
    """
    op = op.upper()
    if op not in ("X", "XT", "XTX", "XXT", "FREQ", "EIGSH", "TO_NUMPY", "FROM_NUMPY"):
        raise ValueError(
            f"Unknown operator {op!r}. Expected one of 'X', 'XT', 'XTX', 'XXT', 'FREQ', 'EIGSH', "
            f"'TO_NUMPY', 'FROM_NUMPY'."
        )
    if backend == "SciPy":
        table = _scipy_operator_table()
    elif backend == "CuPy":
        table = _cupy_operator_table()
    else:
        raise ValueError(f"Unknown backend {backend!r}. Expected 'SciPy' or 'CuPy'.")
    cls = table.get((op, standardized, multi))
    if cls is None:
        raise NotImplementedError(
            f"No {backend} operator for op={op!r}, standardized={standardized}, multi={multi}."
        )
    return cls


class GRGWaitable(ABC):
    """
    Generic interface for a GRG-related job that can be waited upon.
    """

    @abstractmethod
    def result(self) -> Any:
        pass


class GRGScheduler(ABC):
    """
    Generic interface for something that will schedule multiple GRG operations across
    some computation unit (CPU, GPU, ...).
    """

    @abstractmethod
    def submit(
        self, grg: "GRGCalcInterface", operation, *args, **kwargs
    ) -> GRGWaitable:
        pass


class GRGCalcInterface(ABC):
    """
    This is a minimal generic interface for GRG-related calculations. It does not support all GRG features,
    just the ones needed for performing linear algebra-related calculations.
    """

    @property
    @abstractmethod
    def num_samples(self) -> int:
        pass

    @property
    @abstractmethod
    def num_individuals(self) -> int:
        pass

    @property
    @abstractmethod
    def num_mutations(self) -> int:
        pass

    @property
    @abstractmethod
    def ploidy(self) -> int:
        pass

    @property
    @abstractmethod
    def is_phased(self) -> bool:
        pass

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @property
    @abstractmethod
    def has_missing_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def has_individual_coals(self) -> bool:
        pass

    @abstractmethod
    def get_mutation_by_id(self, id: int) -> pygrgl.Mutation:
        pass

    @abstractmethod
    def matmul(
        self,
        input: numpy.typing.NDArray,
        direction: pygrgl.TraversalDirection,
        emit_all_nodes: bool = False,
        by_individual: bool = False,
        init: Optional[Union[str, numpy.typing.NDArray]] = None,
        miss: Optional[numpy.typing.NDArray] = None,
    ):
        pass

    @abstractmethod
    def make_scheduler(
        self, grgs: List["GRGCalcInterface"], workers: int = 1, gated: bool = False
    ):
        pass

    @abstractmethod
    def get_operator(self, op: str, standardized: bool) -> Callable:
        """
        Return the single-GRG LinearOperator class for ``op`` ("X", "XT", "XTX", "XXT"), choosing
        the backend (SciPy/CuPy) appropriate for this calculator. Returns the class, not an
        instance; the caller constructs it (passing direction for "X"/"XT", freqs when standardized).

        ``op="freq"`` and ``op="eigsh"`` instead returns the backend's allele-frequency function, which the caller
        invokes as ``fn(grg, ...)``.
        """
        pass

    @abstractmethod
    def get_multi_operator(self, op: str, standardized: bool) -> Callable:
        """
        Like :meth:`get_operator`, but returns the multi-GRG ("Multi...") LinearOperator class.
        """
        pass

    @abstractmethod
    def device_context(self):
        pass

    @abstractmethod
    def get_raw(self):
        pass

    @abstractmethod
    def _convert_dir(self, d):
        pass


class GRGSeqOp(GRGWaitable):
    def __init__(self, value):
        self._value = value

    def result(self) -> Any:
        return self._value


class GRGSeqSched(GRGScheduler):
    def submit(self, grg: GRGCalcInterface, operation, *args, **kwargs) -> GRGWaitable:
        return GRGSeqOp(operation(*args, **kwargs))


class GRGThreadOp(GRGWaitable):
    def __init__(self, future: concurrent.futures.Future):
        self.future = future

    def result(self) -> Any:
        return self.future.result()


class GRGThreadSched(GRGScheduler):
    def __init__(self, executor: concurrent.futures.Executor):
        self.executor = executor

    def submit(self, grg: GRGCalcInterface, operation, *args, **kwargs) -> GRGWaitable:
        # We don't need the grg object, because the operation encompasses it. Other methods of
        # scheduling may need to do custom swapping of the grg in/out of memory.
        assert isinstance(grg, GRGCalculator)
        return GRGThreadOp(self.executor.submit(operation, *args, **kwargs))


class GRGGatedSched(GRGScheduler):
    def __init__(self, executor: concurrent.futures.Executor, gated=False):
        self.executor = executor
        self.gated = gated
        self._start_gate = threading.Event()
        if not self.gated:
            self._start_gate.set()

    def _gated_operation(self, operation, *args, **kwargs):
        self._start_gate.wait()
        return operation(*args, **kwargs)

    def start(self) -> None:
        self._start_gate.set()

    def reset(self) -> None:
        if not self.gated:
            raise ValueError("Cannot reset a non-gated scheduler")
        self._start_gate.clear()

    def submit(self, grg: GRGCalcInterface, operation, *args, **kwargs) -> GRGWaitable:
        return GRGThreadOp(
            self.executor.submit(self._gated_operation, operation, *args, **kwargs)
        )


class GRGCalculator(GRGCalcInterface):
    """
    Implementaion of the GRG calculator interface for the regular GRG. This is what most
    people will use, and the APIs are agnostic to this: if you pass in a regular GRG to
    the relevant APIs, it will convert it to this for you.
    """

    def __init__(self, grg: pygrgl.GRG):
        self.grg = grg

    @property
    def num_samples(self) -> int:
        return self.grg.num_samples

    @property
    def num_individuals(self) -> int:
        return self.grg.num_individuals

    @property
    def num_mutations(self) -> int:
        return self.grg.num_mutations

    @property
    def ploidy(self) -> int:
        return self.grg.ploidy

    @property
    def is_phased(self) -> bool:
        return self.grg.is_phased

    @property
    def num_nodes(self) -> int:
        return self.grg.num_nodes

    @property
    def num_edges(self) -> int:
        return self.grg.num_edges

    @property
    def has_missing_data(self) -> bool:
        return self.grg.has_missing_data

    @property
    def has_individual_coals(self) -> bool:
        return self.grg.has_individual_coals

    def get_mutation_by_id(self, id: int) -> pygrgl.Mutation:
        return self.grg.get_mutation_by_id(id)

    def matmul(
        self,
        input: numpy.typing.NDArray,
        direction: pygrgl.TraversalDirection,
        emit_all_nodes: bool = False,
        by_individual: bool = False,
        init: Optional[Union[str, numpy.typing.NDArray]] = None,
        miss: Optional[numpy.typing.NDArray] = None,
    ):
        return pygrgl.matmul(
            self.grg,
            input,
            direction,
            emit_all_nodes=emit_all_nodes,
            by_individual=by_individual,
            init=init,
            miss=miss,
        )

    def make_scheduler(
        self, grgs: List["GRGCalcInterface"], workers: int = 1, gated: bool = False
    ):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        return GRGThreadSched(executor)

    def get_operator(self, op: str, standardized: bool) -> Callable:
        return _select_operator_cls("SciPy", op, standardized, multi=False)

    def get_multi_operator(self, op: str, standardized: bool) -> Callable:
        return _select_operator_cls("SciPy", op, standardized, multi=True)

    def device_context(self):
        return contextlib.suppress()

    def get_raw(self):
        return self.grg

    def _convert_dir(self, d):
        return d


class GRGSpMVCalculator(GRGCalcInterface):
    """
    Implementaion of the GRG calculator interface for the SPMV-based GRG.
    """

    def __init__(self, grg_spmv):
        self._op = grg_spmv

    @property
    def device(self) -> int | None:
        return getattr(self._op, "device", None)

    @property
    def use_cupy(self) -> bool:
        return getattr(self._op, "use_cupy", False)

    @property
    def num_samples(self) -> int:
        return self._op.num_samples

    @property
    def num_individuals(self) -> int:
        return self._op.num_individuals

    @property
    def num_mutations(self) -> int:
        return self._op.num_mutations

    @property
    def ploidy(self) -> int:
        return self._op.ploidy

    @property
    def is_phased(self) -> bool:
        raise NotImplementedError

    @property
    def num_nodes(self) -> int:
        return self._op.num_nodes

    @property
    def num_edges(self) -> int:
        return self._op.num_edges

    @property
    def has_missing_data(self) -> bool:
        return self._op.has_missing_data

    @property
    def has_individual_coals(self) -> bool:
        return True  # TODO: support this downstream

    def get_mutation_by_id(self, id: int) -> pygrgl.Mutation:
        return self._op.get_mutation_by_id(id)

    def _convert_dir(self, d: pygrgl.TraversalDirection):
        if d == pygrgl.TraversalDirection.DOWN:
            return "down"
        else:
            assert d == pygrgl.TraversalDirection.UP
            return "up"

    def matmul(
        self,
        input: numpy.typing.NDArray,
        direction: pygrgl.TraversalDirection,
        emit_all_nodes: bool = False,
        by_individual: bool = False,
        init: Optional[Union[str, numpy.typing.NDArray]] = None,
        miss: Optional[numpy.typing.NDArray] = None,
    ) -> numpy.typing.NDArray:
        if self.use_cupy:
            with cupy.cuda.Device(self.device):
                mm_input = cupy.asarray(input)
                mm_init = cupy.asarray(init) if init is not None else init
                mm_miss = cupy.asarray(miss) if miss is not None else miss
                result = self._op.matmul(
                    mm_input,
                    self._convert_dir(direction),
                    emit_all_nodes=emit_all_nodes,
                    by_individual=by_individual,
                    init=mm_init,
                    miss=mm_miss,
                )
                cupy.cuda.Device().synchronize()
                result = cupy.asnumpy(result)
        else:
            result = self._op.matmul(
                input,
                self._convert_dir(direction),
                emit_all_nodes=emit_all_nodes,
                by_individual=by_individual,
                init=init,
                miss=miss,
            )
        return result

    def get_raw(self):
        return self._op

    def make_scheduler(
        self, grgs: List["GRGCalcInterface"], workers: int = 1, gated: bool = False
    ):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        return GRGGatedSched(executor=executor, gated=gated)

    def get_operator(self, op: str, standardized: bool) -> Callable:
        backend = "CuPy" if self.use_cupy else "SciPy"
        return _select_operator_cls(backend, op, standardized, multi=False)

    def get_multi_operator(self, op: str, standardized: bool) -> Callable:
        backend = "CuPy" if self.use_cupy else "SciPy"
        return _select_operator_cls(backend, op, standardized, multi=True)

    def device_context(self):
        return cupy.cuda.Device(self.device) if self.use_cupy else contextlib.suppress()


def load_grg_calculator(filename: str) -> GRGCalcInterface:
    """
    Load a file as one of the supported GRG calculator file types.
    """
    extension_to_loader: Dict[str, Callable[[str], GRGCalcInterface]] = {
        ".grg": (
            lambda filename: GRGCalculator(
                pygrgl.load_immutable_grg(filename, load_up_edges=False)
            )
        ),
    }
    from grapp.util.exceptions import UserInputError

    if pygrgl_spmv is not None:

        def _raise_spmv_error(filename: str) -> GRGCalcInterface:
            raise UserInputError("grg_spmv files cannot be loaded directly.")

        extension_to_loader[".grg_spmv"] = _raise_spmv_error
    for ext, loader in extension_to_loader.items():
        if filename.endswith(ext):
            return loader(filename)
    raise UserInputError(
        f"Only the following file extensions are supported: {', '.join(extension_to_loader.keys())}"
    )


# Internal method: if we get to a place where we need GRGCalcInterface, then this ensures that
# we have one.
def _wrap_grg(grg: Union[pygrgl.GRG, GRGCalcInterface]) -> GRGCalcInterface:
    if isinstance(grg, pygrgl.GRG):
        return GRGCalculator(grg)
    assert isinstance(grg, GRGCalcInterface)
    return grg
