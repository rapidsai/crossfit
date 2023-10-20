import numpy as np
import pytest
import torch

from crossfit.data import convert_array


@pytest.mark.parametrize("array", [torch.asarray, np.array])
def test_simple_convert(array):
    tensor = convert_array(array([1, 2, 3]), torch.Tensor)

    assert isinstance(tensor, torch.Tensor)
