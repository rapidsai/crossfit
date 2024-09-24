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

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import cudf
import cupy as cp
from cudf.core.column import as_column
from cudf.core.dtypes import ListDtype
from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from cudf.core.buffer import Buffer
    from cudf.core.column import ColumnBase
    from cudf.core.column.numerical import NumericalColumn


@lru_cache
def _is_cudf_gte_24_10():
    current_cudf_version = parse_version(cudf.__version__)
    cudf_24_10_version = parse_version("24.10.0")

    if current_cudf_version >= cudf_24_10_version or (
        current_cudf_version.base_version >= "24.10.0" and current_cudf_version.is_prerelease
    ):
        return True
    elif current_cudf_version < cudf_24_10_version or (
        current_cudf_version.base_version < "24.10.0" and current_cudf_version.is_prerelease
    ):
        return False
    else:
        msg = f"Found uncaught cudf version {current_cudf_version}"
        raise NotImplementedError(msg)


def _construct_series_from_list_column(index: Any, lc: cudf.core.column.ListColumn) -> cudf.Series:
    if not _is_cudf_gte_24_10():
        return cudf.Series(data=lc, index=index)
    else:
        from cudf.core.index import ensure_index

        return cudf.Series._from_column(column=lc, index=ensure_index(index))


def _construct_list_column(
    size: int,
    dtype: ListDtype,
    mask: Optional["Buffer"] = None,
    offset: int = 0,
    null_count: Optional[int] = None,
    children: tuple["NumericalColumn", "ColumnBase"] = (),  # type: ignore[assignment]
) -> cudf.core.column.ListColumn:
    kwargs = dict(
        size=size,
        dtype=dtype,
        mask=mask,
        offset=offset,
        null_count=null_count,
        children=children,
    )

    if not _is_cudf_gte_24_10():
        return cudf.core.column.ListColumn(**kwargs)
    else:
        # in 24.10 ListColumn added `data` kwarg see https://github.com/rapidsai/crossfit/issues/84
        return cudf.core.column.ListColumn(data=None, **kwargs)


def create_list_series_from_1d_or_2d_ar(ar, index):
    """
    Create a cudf list series  from 2d arrays
    """
    if len(ar.shape) == 1:
        n_rows, *_ = ar.shape
        n_cols = 1
    elif len(ar.shape) == 2:
        n_rows, n_cols = ar.shape
    else:
        return RuntimeError(f"Unexpected input shape: {ar.shape}")
    data = as_column(ar.flatten())
    offset_col = as_column(cp.arange(start=0, stop=len(data) + 1, step=n_cols), dtype="int32")
    mask_col = cp.full(shape=n_rows, fill_value=cp.bool_(True))
    mask = cudf._lib.transform.bools_to_mask(as_column(mask_col))

    lc = _construct_list_column(
        size=n_rows,
        dtype=cudf.ListDtype(data.dtype),
        mask=mask,
        offset=0,
        null_count=0,
        children=(offset_col, data),
    )
    return _construct_series_from_list_column(lc=lc, index=index)


def create_nested_list_series_from_3d_ar(ar, index):
    """
    Create a cudf list of lists series from 3d arrays
    """
    n_slices, n_rows, n_cols = ar.shape
    flattened_data = ar.reshape(-1)  # Flatten the 3-D array into 1-D

    # Inner list offsets (for each row in 2D slices)
    inner_offsets = cp.arange(
        start=0, stop=n_cols * n_rows * n_slices + 1, step=n_cols, dtype="int32"
    )
    inner_list_data = as_column(flattened_data)
    inner_list_offsets = as_column(inner_offsets)

    # Outer list offsets (for each 2D slice in the 3D array)
    outer_offsets = cp.arange(start=0, stop=n_slices + 1, step=1, dtype="int32") * n_rows
    outer_list_offsets = as_column(outer_offsets)

    # Constructing the nested ListColumn
    inner_lc = _construct_list_column(
        size=inner_offsets.size - 1,
        dtype=cudf.ListDtype(inner_list_data.dtype),
        children=(inner_list_offsets, inner_list_data),
        mask=None,
        offset=0,
        null_count=None,
    )

    lc = _construct_list_column(
        size=n_slices,
        dtype=cudf.ListDtype(inner_list_data.dtype),
        children=(outer_list_offsets, inner_lc),
        mask=None,
        offset=0,
        null_count=None,
    )

    return _construct_series_from_list_column(lc=lc, index=index)
