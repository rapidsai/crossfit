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
import torch

from crossfit.data import convert_array


@pytest.mark.parametrize("array", [torch.asarray, np.array])
def test_simple_convert(array):
    tensor = convert_array(array([1, 2, 3]), torch.Tensor)

    assert isinstance(tensor, torch.Tensor)
