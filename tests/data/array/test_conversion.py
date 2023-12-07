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

from crossfit.data import convert_array


def test_convert_no_op():
    array = np.array([1, 2, 3])

    assert np.all(array == convert_array(array, np.ndarray))


@pytest.mark.parametrize("to_type", convert_array.supports[np.ndarray])
def test_convert_roundtrip(to_type):
    from_array = np.array([1, 2, 3])
    converted = convert_array(from_array, to_type)
    assert isinstance(converted, to_type)

    orig = convert_array(converted, np.ndarray)
    assert type(orig) is np.ndarray

    assert np.all(from_array == orig)
