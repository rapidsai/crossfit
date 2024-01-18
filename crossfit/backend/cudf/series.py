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

import cudf
import cupy as cp
from cudf.core.column import as_column


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
    mask_col = cp.full(shape=n_rows, fill_value=True)
    mask = cudf._lib.transform.bools_to_mask(as_column(mask_col))
    lc = cudf.core.column.ListColumn(
        size=n_rows,
        dtype=cudf.ListDtype(data.dtype),
        mask=mask,
        offset=0,
        null_count=0,
        children=(offset_col, data),
    )

    return cudf.Series(lc, index=index)


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
    lc = cudf.core.column.ListColumn(
        size=n_slices,
        dtype=cudf.ListDtype(inner_list_data.dtype),
        children=(
            outer_list_offsets,
            cudf.core.column.ListColumn(
                size=inner_offsets.size - 1,
                dtype=cudf.ListDtype(inner_list_data.dtype),
                children=(inner_list_offsets, inner_list_data),
            ),
        ),
    )

    return cudf.Series(lc, index=index)
