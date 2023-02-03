import numpy as np
from sklearn.metrics import accuracy_score

from crossfit.metrics.mean import Mean, create_mean_metric


def test_simple_mean():
    # Generate random array
    arr = np.random.rand(1000)

    mean = Mean()(arr, axis=0)
    assert isinstance(mean, Mean)
    assert mean.result == np.mean(arr)


def test_simple_accuracy():
    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000) > 0.5

    acc = Mean(pre=accuracy_score)(y_true, y_pred)
    assert isinstance(acc, Mean)
    assert acc.result == accuracy_score(y_true, y_pred)

    accuracy = create_mean_metric(accuracy_score)
    assert accuracy(y_true, y_pred).result == accuracy_score(y_true, y_pred)
