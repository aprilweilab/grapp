from abc import ABC, abstractmethod
import pygrgl
import numpy
import concurrent.futures
from typing import Optional, Union, Dict, Callable, List, Any
from grapp.util.exceptions import UserInputError


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
    def make_scheduler(self, grgs: List["GRGCalcInterface"], workers: int = 1):
        pass


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

    def make_scheduler(self, grgs: List["GRGCalcInterface"], workers: int = 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        return GRGThreadSched(executor)


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
