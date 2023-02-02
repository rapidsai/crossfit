import pytest

import numpy as np
import torch

import crossfit.array as cnp


@pytest.mark.parametrize("array", [torch.asarray, np.array])
def test_simple_convert(array):
    tensor = cnp.convert(array([1, 2, 3]), torch.Tensor)

    assert isinstance(tensor, torch.Tensor)
