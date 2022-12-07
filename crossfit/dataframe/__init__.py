from typing import Literal

from crossfit.dataframe import pandas_backend
from crossfit.dataframe import cudf_backend
from crossfit.utils.df_utils import requires_df_backend


BackendName = Literal["pandas", "cpu", "cudf", "gpu"]


def df_backend(name_or_obj: BackendName = "pandas"):
    if not isinstance(name_or_obj, str):
        name = name_or_obj.__module__.split(".")[0]
    else:
        name = name_or_obj

    if name is None:
        raise ValueError("name is None")
    if not isinstance(name, str):
        raise ValueError(f"name is not a string: {name}")

    if name in {"pandas", "cpu"}:
        return pandas_backend
    if name in {"cudf", "gpu"}:
        if not cudf_backend.is_installed:
            raise ValueError("cudf is not installed")
        return cudf_backend

    raise ValueError(f"Unknown library name: {name}")


__all__ = [
    "df_backend",
    "requires_df_backend",
    "pandas_backend",
    "cudf_backend",
    "BackendName",
]
