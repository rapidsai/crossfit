import pytest
from collections.abc import Mapping

import pandas as pd

try:
    import cudf

except ImportError:
    cudf = None

CUDF_MARK = pytest.mark.skipif(cudf is None, reason="cudf not found")


def to_list(obj):
    """Convert Array or Series to list"""
    if hasattr(obj, "to_pandas"):
        # cudf -> pandas
        return to_list(obj.to_pandas())
    elif hasattr(obj, "to_list"):
        # pd.Series to list
        return obj.to_list()
    elif hasattr(obj, "tolist"):
        # np/cupt.ndarray -> list
        return obj.tolist()
    return list(obj)


def is_leaf_node_instance_of(d, cls):
    if not isinstance(d, Mapping):
        return isinstance(d, cls)
    return all(is_leaf_node_instance_of(v, cls) for v in d.values())


def sample_df(data: dict):
    """DataFrame-backend parameterization

    Converts dictionary of data to pandas- and cudf-backed data
    """
    return pytest.mark.parametrize(
        "df",
        [
            pytest.param(pd.DataFrame(data), id="pandas"),
            pytest.param(
                cudf.DataFrame(data) if cudf else None,
                id="cudf",
                marks=CUDF_MARK,
            ),
        ],
    )
