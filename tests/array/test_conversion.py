import pytest

import numpy as np

import crossfit.array as cnp


def test_convert_no_op():
    array = np.array([1, 2, 3])

    assert cnp.all(array == cnp.convert(array, np.ndarray))


@pytest.mark.parametrize("to_type", cnp.convert.supports[np.ndarray])
def test_convert_roundtrip(to_type):
    from_array = np.array([1, 2, 3])
    converted = cnp.convert(from_array, to_type)
    assert isinstance(converted, to_type)

    orig = cnp.convert(converted, np.ndarray)
    assert type(orig) == np.ndarray

    assert cnp.all(from_array == orig)
