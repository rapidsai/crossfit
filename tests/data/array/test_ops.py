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

from crossfit.data import numpy as cnp

left = np.array([1, 2, 3])
right = np.array([4, 5, 6])


def test_minimum():
    added = cnp.minimum(left, right)
    assert np.minimum(left, right).all() == added.all()


def test_maximum():
    added = cnp.maximum(left, right)
    assert np.maximum(left, right).all() == added.all()


def test_sum():
    added = sum(left, right)
    assert (left + right).all() == added.all()
