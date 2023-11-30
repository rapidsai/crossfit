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


def create_list_series_from_2d_ar(ar, index):
    """
    Create a cudf list series  from 2d arrays
    """
    n_rows, n_cols = ar.shape
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
