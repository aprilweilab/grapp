import numpy
import pandas
import pygrgl
from typing import Optional, List, TextIO, Union
from grapp.util.exceptions import UserInputError


def numpy_to_tsv(
    file_obj: TextIO,
    matrix: numpy.typing.NDArray,
    column_names: Optional[List[str]] = None,
):
    SEP = "\t"
    assert matrix.ndim == 2
    if column_names is not None:
        assert len(column_names) == matrix.shape[1]
        print(SEP.join(column_names), file=file_obj)
    for row in matrix:
        print(SEP.join(map(str, row)), file=file_obj)


def pandas_to_tsv(
    file_obj: Union[TextIO, str],
    dataframe: pandas.DataFrame,
):
    dataframe.to_csv(file_obj, sep="\t", index=False)


def load_immutable(filename: str, load_up_edges: bool = False) -> pygrgl.GRG:
    if not filename.endswith(".grg"):
        raise UserInputError("Only .grg files are supported for this operation.")
    return pygrgl.load_immutable_grg(filename, load_up_edges=load_up_edges)
