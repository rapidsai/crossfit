import numpy as np
from sklearn.metrics import roc_auc_score

from crossfit.ml.classification import BinaryMetrics


def test_auc_calculation():
    # Generate some random data
    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000)

    # Calculate the AUC using numpy
    tp = np.sum((y_true == 1) & (y_pred > 0.5))
    tn = np.sum((y_true == 0) & (y_pred <= 0.5))
    fp = np.sum((y_true == 0) & (y_pred > 0.5))
    fn = np.sum((y_true == 1) & (y_pred <= 0.5))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    auc_numpy = np.trapz([tpr, fpr], [0, 1])

    state = BinaryMetrics()(y_pred, y_true)
    assert state.auc == auc_numpy

    # Calculate the AUC using scikit-learn
    auc_sklearn = roc_auc_score(y_true, y_pred)

    # Ensure that the two AUC values are close
    assert abs(auc_numpy - auc_sklearn) < 1e-1
