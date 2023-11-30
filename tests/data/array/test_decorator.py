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
import pytest

from crossfit.data import crossarray
from crossfit.utils import test_utils


def max_test(x, y):
    return np.maximum(x, y)


def nesting_test(x, y):
    return test_utils.min_test(x, y) + max_test(x, y)


@pytest.mark.parametrize("fn", [np.all, np.sum, np.mean, np.std, np.var, np.any, np.prod])
def test_simple_numpy_function_crossnp(fn):
    crossfn = crossarray(fn)

    x = np.array([1, 2, 3])

    _cross_out = crossfn(x)
    _np_out = fn(x)

    assert np.all(_cross_out == _np_out)


@pytest.mark.parametrize("fn", [np.minimum, np.maximum, max_test, test_utils.min_test])
def test_combine_numpy_function_crossnp(fn):
    crossfn = crossarray(fn)

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    _cross_out = crossfn(x, y)
    _np_out = fn(x, y)

    assert np.all(_cross_out == _np_out)
