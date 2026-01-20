import numpy
import pandas
from typing import Optional, List, TextIO, Union


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
