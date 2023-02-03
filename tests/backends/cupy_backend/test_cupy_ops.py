import cupy as cp
import numpy as np

from crossfit.data import convert_array, crossarray


@crossarray
def test_cupy_backend():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    cp_min = np.minimum(cp.array(arr1), cp.array(arr2))
    np_min = np.minimum(np.array(arr1), np.array(arr2))

    assert np.all(cp_min == convert_array(np_min, cp.ndarray))
