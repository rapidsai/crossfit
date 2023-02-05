import pytest

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import crossfit as cf
from crossfit.backends.tf import from_tf_metric


y_true = np.random.randint(2, size=1000)
y_pred = np.random.rand(1000)
country = np.random.choice(["US", "UK", "FR"], size=1000)


@pytest.mark.skip("TODO: fix this test on GH-actions")
def test_tf_accuracy():
    acc = from_tf_metric(tf.keras.metrics.Accuracy())

    state = acc.prepare(y_true, y_pred > 0.5)
    np.testing.assert_almost_equal(state.result, accuracy_score(y_true, y_pred > 0.5))


@pytest.mark.skip("TODO: fix this test on GH-actions")
def test_tf_accuracy_dask():
    import dask.dataframe as dd
    from crossfit.backends.dask.aggregate import aggregate

    acc = from_tf_metric(tf.keras.metrics.BinaryAccuracy())
    precision = from_tf_metric(tf.keras.metrics.Precision())
    recall = from_tf_metric(tf.keras.metrics.Recall())

    aggs = {"accuracy": acc, "precision": precision, "recall": recall}

    metrics = cf.Aggregator(aggs, pre=lambda x: (x["targets"], x["predictions"]))
    grouped_metrics = cf.Aggregator(metrics, groupby="country")

    df = pd.DataFrame({"targets": y_true, "predictions": y_pred, "country": country})
    ddf = dd.from_pandas(df, npartitions=2)

    aggregated = aggregate(ddf, metrics, to_frame=True)
    assert isinstance(aggregated, pd.DataFrame)
    assert set(aggregated.columns) == {"accuracy", "precision", "recall"}

    per_country = aggregate(ddf, grouped_metrics, to_frame=True)
    assert isinstance(per_country, pd.DataFrame)
