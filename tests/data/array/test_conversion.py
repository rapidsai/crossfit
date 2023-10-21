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
    assert type(orig) == np.ndarray

    assert np.all(from_array == orig)
