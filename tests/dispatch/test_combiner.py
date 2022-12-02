import numpy as np

from crossfit.dispatch.combiner import Max, Min, Sum

left = np.array([1, 2, 3])
right = np.array([4, 5, 6])


def test_min():
    added = Min.combine(left, right)
    assert np.minimum(left, right).all() == added.all()


def test_max():
    added = Max.combine(left, right)
    assert np.maximum(left, right).all() == added.all()


def test_sum():
    added = Sum.combine(left, right)
    assert (left + right).all() == added.all()
