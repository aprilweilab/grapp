import numpy
from typing import Optional, List, TextIO


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
