import cupy as cp
import numpy as np

import crossfit.array as cnp
from crossfit.array import convert


def test_cupy_backend():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    cp_min = cnp.minimum(cp.array(arr1), cp.array(arr2))
    np_min = np.minimum(np.array(arr1), np.array(arr2))

    assert cnp.all(cp_min == convert(np_min, cp.ndarray))
