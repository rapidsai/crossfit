import pytest

import numpy as np
import tensorflow as tf

# from crossfit.ml.classification import BinaryClassificationState
from crossfit.backends.tf.metrics import to_tf_metric
from crossfit.metrics import create_mean_metric


def accuracy_score(y_true, y_pred, sample_weight=None):
    return np.mean(y_true == y_pred)


# class BinaryMetrics(cf.Aggregator):
#     state = BinaryClassificationState

#     def prepare(self, labels, predictions, sample_weight=None):
#         predictions = np.array(predictions > 0.5, np.int32)
#         # Calculate true-positives
#         tp = np.sum((predictions == 1) & (labels == 1))

#         # Calculate true-negatives
#         tn = np.sum((predictions == 0) & (labels == 0))

#         # Calculate false-positives
#         fp = np.sum((predictions == 1) & (labels == 0))

#         # Calculate false-negatives
#         fn = np.sum((predictions == 0) & (labels == 1))

#         return BinaryClassificationState(tp, tn, fp, fn)

#     def present(self, state: BinaryClassificationState):
#         # Somehow a keras metric is not allowed to return a dict
#         #   See: https://github.com/keras-team/keras/issues/16665
#         return {
#             # "accuracy": state.accuracy,
#             # "precision": state.precision,
#             # "recall": state.recall,
#             # "f1": state.f1,
#             "auc": state.auc,
#         }


@pytest.mark.parametrize("jit_compile", [True, False])
def test_to_tf_metric(jit_compile):
    accuracy = create_mean_metric(accuracy_score)
    metric = to_tf_metric(accuracy)

    # generate some random tensors
    preds = tf.random.uniform((10,))
    targets = tf.random.uniform((10,), maxval=2, dtype=tf.int32)

    results = metric(targets, preds > 0.5)
    assert isinstance(results, tf.Tensor)
