import pytest

import tensorflow as tf
import numpy as np

from crossfit.utils import test_utils
from crossfit.data import crossarray, convert_array


arr1 = [1, 2, 3]
arr2 = [4, 5, 6]


def max_test(x, y):
    return np.maximum(x, y)


def nested(x, y):
    return max_test(x, y) + test_utils.min_test(x, y)


@pytest.mark.parametrize("jit", [True, False])
def test_tf_backend(jit):
    minimum = crossarray(np.minimum, jit=jit)
    tf_min = minimum(tf.constant(arr1), tf.constant(arr2))
    np_min = minimum(np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32))

    with crossarray:
        assert np.all(tf_min == convert_array(np_min, tf.Tensor))


def test_tf_crossnp():
    cross_max = crossarray(max_test)
    tf_out = cross_max(tf.constant(arr1), tf.constant(arr2))
    np_out = cross_max(np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32))

    with crossarray:
        assert np.all((tf_out == convert_array(np_out, tf.Tensor)).numpy())


# TODO: Fix jit compilation of nested functions in TF
#   We might have to bring back the ast approach for that?
@pytest.mark.parametrize("jit", [False])
def test_tf_crossnp_nested(jit):
    cross_nested = crossarray(nested, jit=jit)
    tf_out = cross_nested(tf.constant(arr1), tf.constant(arr2))
    np_out = cross_nested(
        np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32)
    )

    with crossarray:
        assert np.all((tf_out == convert_array(np_out, tf.Tensor)).numpy())
