# Copyright 2025 NVIDIA CORPORATION
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
import pylibcudf as plc


def create_list_series_from_1d_or_2d_ar(ar, index):
    """
    Create a cudf list series  from 2d arrays
    """
    arr = cp.asarray(ar)
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    if not isinstance(index, (cudf.RangeIndex, cudf.Index, cudf.MultiIndex)):
        index = cudf.Index(index)
    return cudf.Series.from_pylibcudf(
        plc.Column.from_cuda_array_interface(arr),
        metadata={"index": index},
    )


def create_nested_list_series_from_3d_ar(ar, index):
    """
    Create a cudf list of lists series from 3d arrays
    """
    arr = cp.asarray(ar)
    if not isinstance(index, (cudf.RangeIndex, cudf.Index, cudf.MultiIndex)):
        index = cudf.Index(index)
    return cudf.Series.from_pylibcudf(
        plc.Column.from_cuda_array_interface(arr),
        metadata={"index": index},
    )
