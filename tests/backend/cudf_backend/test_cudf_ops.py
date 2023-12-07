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
import numpy as np

from crossfit.data import convert_array, crossarray


@crossarray
def test_cudf_backend():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    cp_min = np.minimum(cudf.Series(arr1).values, cudf.Series(arr2).values)
    np_min = np.minimum(np.array(arr1), np.array(arr2))

    assert np.all(cp_min == convert_array(np_min, cudf.Series))
