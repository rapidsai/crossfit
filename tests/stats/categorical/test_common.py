import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.categorical.common import ValueCounts, ValueCountsState

df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.randn(10),
        "c": np.random.randn(10),
    }
)


def test_value_counts_calculate():
    mf: MetricFrame = calculate_per_col(ValueCounts(), df[["a", "a2"]])

    assert isinstance(mf, MetricFrame)
    assert len(mf.state_df) == 2

    result = mf.result()
    assert isinstance(result, pd.DataFrame)


def test_value_counts_combine():
    arr = df["a"]

    state = ValueCounts().prepare(arr)
    combined = state + state

    assert isinstance(combined, ValueCountsState)
    assert np.all(combined.values == np.array([0, 1, 2, 3, 4]))
    assert np.all(combined.counts == 4)
