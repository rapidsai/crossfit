# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping

import pandas as pd
import pytest

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
