import pytest

import tensorflow as tf
import numpy as np

from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils._array_api import get_namespace

import crossfit.array as cnp
from crossfit.array import convert
from crossfit.utils import test_utils
from crossfit.array.backend.tf_backend import TFBackend


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


@pytest.mark.parametrize(
    "metric",
    [
        "mean_squared_error",
        "median_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
    ],
)
def test_tf_crossnp_sklearn_regression(metric):
    metric = cnp.crossnp(getattr(metrics, metric))

    tf_out = metric(
        tf.constant(arr1, dtype=tf.float32), tf.constant(arr2, dtype=tf.float32)
    )
    np_out = metric(np.array(arr1, dtype=np.float32), np.array(arr2, dtype=np.float32))

    assert tf_out == np_out


def test_tf_crossnp_type_of_target():
    tot = cnp.crossnp(type_of_target)
    con = [0.1, 0.6]

    tensor = tf.constant(con)
    namespace, _ = get_namespace(tensor)
    assert isinstance(namespace._namespace, TFBackend)

    assert tot(tf.constant(con)) == tot(con)


@pytest.mark.parametrize(
    "metric",
    [metrics.accuracy_score, metrics.precision_score, metrics.recall_score],
)
def test_tf_crossnp_sklearn_clf(metric):

    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000)

    # cross = cnp.crossnp(metric, validate_array_type=tf.Tensor)
    cross = cnp.crossnp(metric)

    tf_out = cross(tf.constant(y_true), tf.constant(y_pred) > 0.5)
    np_out = cross(y_true, y_pred > 0.5)

    assert tf_out == np_out
