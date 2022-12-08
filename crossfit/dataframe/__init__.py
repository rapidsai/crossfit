from typing import Literal, Protocol, runtime_checkable, Optional, Union

from crossfit.dataframe import pandas_backend
from crossfit.dataframe import cudf_backend
from crossfit.utils.df_utils import requires_df_backend


BackendName = Literal["pandas", "cpu", "cudf", "gpu"]


@runtime_checkable
class Backend(Protocol):
    def is_grouped(self, df) -> bool:
        ...


def df_backend(
    name_or_obj: BackendName = "pandas", protocol: Optional[Backend] = None
) -> Backend:
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


def test_function(a: int = 1) -> Union[int, str]:
    if a == 1:
        return 1

    return str(a)


__all__ = [
    "df_backend",
    "requires_df_backend",
    "pandas_backend",
    "cudf_backend",
    "BackendName",
]


if __name__ == "__main__":
    # cd = df_backend("cudf")

    reveal_type(test_function(1))
