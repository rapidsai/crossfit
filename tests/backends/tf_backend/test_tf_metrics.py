import pytest

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from crossfit.backends.tf import from_tf_metric


@pytest.skip("TODO: fix this test on GH-actions")
def test_tf_accuracy():
    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000)

    acc = from_tf_metric(tf.keras.metrics.Accuracy())

    state = acc.prepare(y_true, y_pred > 0.5)
    np.testing.assert_almost_equal(state.result, accuracy_score(y_true, y_pred > 0.5))
