import tensorflow as tf
import numpy as np

import crossfit.array as cnp
from crossfit.array import convert
from crossfit.utils import test_utils


arr1 = [1, 2, 3]
arr2 = [4, 5, 6]


def max_test(x, y):
    return np.maximum(x, y)


def nested(x, y):
    return max_test(x, y) + test_utils.min_test(x, y)


def test_tf_backend():
    tf_min = cnp.minimum(tf.constant(arr1), tf.constant(arr2))
    np_min = np.minimum(np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32))

    assert cnp.all(tf_min == convert(np_min, tf.Tensor))


def test_tf_crossnp():
    cross_max = cnp.crossnp(max_test)
    tf_out = cross_max(tf.constant(arr1), tf.constant(arr2))
    np_out = cross_max(np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32))

    assert cnp.all((tf_out == convert(np_out, tf.Tensor)).numpy())


def test_tf_crossnp_nested():
    cross_nested = cnp.crossnp(nested)
    tf_out = cross_nested(tf.constant(arr1), tf.constant(arr2))
    np_out = cross_nested(
        np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32)
    )

    assert cnp.all((tf_out == convert(np_out, tf.Tensor)).numpy())
