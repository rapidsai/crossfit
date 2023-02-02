import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate
from crossfit.ml.classification import BinaryMetrics
from crossfit.ml.metrics import tf_metrics as tfm


def test_tf_accuracy():
    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000)

    state = tfm.TFAccuracy()(y_pred > 0.5, y_true)
    np.testing.assert_almost_equal(
        state.result, BinaryMetrics().prepare(y_pred, y_true).accuracy
    )
    state_df = calculate(tfm.TFAccuracy(), y_true, y_pred)
    assert isinstance(state_df.result(), pd.DataFrame)
