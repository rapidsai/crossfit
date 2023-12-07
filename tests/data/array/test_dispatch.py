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

import numpy as np

import crossfit as cf


def custom_function(a, b):
    # Custom function that uses numpy and returns
    # different results when monkey-patched

    c = np.add(a, b)

    if hasattr(np, "__origdict__"):
        c += 1

    return c


def test_monkey_path_np():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    from sklearn import metrics

    # Call the custom function within the context manager
    with cf.crossarray:
        x = np.array(arr1)
        y = np.array(arr2)
        z = custom_function(x, y)
        met = metrics.mean_squared_error(x, y)

    assert not getattr(np, "__origdict__", None)
    assert np.all(met == metrics.mean_squared_error(arr1, arr2))
    assert np.all(z == custom_function(arr1, arr2) + 1)
