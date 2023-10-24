import numpy as np
import pytest

from crossfit.data import crossarray
from crossfit.utils import test_utils


def max_test(x, y):
    return np.maximum(x, y)


def nesting_test(x, y):
    return test_utils.min_test(x, y) + max_test(x, y)


@pytest.mark.parametrize(
    "fn", [np.all, np.sum, np.mean, np.std, np.var, np.any, np.prod]
)
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
