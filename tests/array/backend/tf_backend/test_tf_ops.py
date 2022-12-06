import tensorflow as tf
import numpy as np

import crossfit.array as cnp
from crossfit.array import convert


def test_tf_backend():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    tf_min = cnp.minimum(tf.constant(arr1), tf.constant(arr2))
    np_min = np.minimum(np.array(arr1, dtype=np.int32), np.array(arr2, dtype=np.int32))

    assert cnp.all(tf_min == convert(np_min, tf.Tensor))
