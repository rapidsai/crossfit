import pytest

import pandas as pd

from crossfit.dataframe import df_backend, cudf_backend, pandas_backend, Backend

cpd = df_backend()
data = {"a": [1, 2, 3], "b": [1, 2, 3]}


def test_pandas_backend():
    assert cpd == pandas_backend
    assert cpd.is_grouped(pd.DataFrame(data).groupby("a"))


def test_cudf_backend():
    try:
        import cudf
    except ImportError:
        cudf = None

    if cudf is None:
        with pytest.raises(ValueError) as exc_info:
            lib = df_backend("cudf")
            assert "cudf" in str(exc_info.value)
    else:
        lib = df_backend("cudf")
        df = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

        assert df_backend(df) == cudf_backend
        assert df_backend(df.groupby("a")) == cudf_backend
        assert lib.is_grouped(df.groupby("a"))
        assert lib == cudf_backend
