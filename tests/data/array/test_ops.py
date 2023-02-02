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
