import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from crossfit.ml.classification import BinaryMetrics, BinaryMetricsThresholded, BinClfThresholdedState

# Generate some random data
y_true = np.random.randint(2, size=1000)
y_pred = np.random.rand(1000)


def test_auc_calculation():
    state = BinaryMetricsThresholded(num_thresholds=1000)(y_pred, y_true)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Calculate the AUC using scikit-learn
    auc_sklearn = roc_auc_score(y_true, y_pred)

    # Ensure that the two AUC values are close
    assert abs(state.auc - auc_sklearn) < 1e-1


def test_thresholded_bin_clf():
    split_point = 100
    state = BinaryMetricsThresholded()(y_pred[:split_point], y_true[:split_point])
    state_2 = BinaryMetricsThresholded()(y_pred[split_point:], y_true[split_point:])
    
    merged = state + state_2
    assert isinstance(merged, BinClfThresholdedState)
    assert len(merged.fn) == len(state.fn)
    
    state_thresholded = BinaryMetricsThresholded(num_thresholds=10)(y_pred, y_true)
    merged_2 = state + state_thresholded
    
    assert isinstance(merged_2, BinClfThresholdedState)
    assert len(merged_2.fn) == len(state.fn)
