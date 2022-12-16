import numpy as np

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.categorical.common import ValueCounts, ValueCountsState
from tests.utils import sample_df, to_list

data = {
    "a": list(range(5)) * 2,
    "a2": list(range(5)) * 2,
    "b": np.random.randn(10),
    "c": np.random.randn(10),
}


@sample_df(data)
def test_value_counts_calculate(df):
    mf: MetricFrame = calculate_per_col(ValueCounts(), df[["a", "a2"]])

    assert isinstance(mf, MetricFrame)
    assert len(mf.state_df) == 2

    result = mf.result()
    assert isinstance(result, type(df))


@sample_df(data)
def test_value_counts_combine(df):
    arr = df["a"]

    state = ValueCounts().prepare(arr)
    combined = state + state

    assert isinstance(combined, ValueCountsState)
    assert to_list(combined.values) == [0, 1, 2, 3, 4]
    assert to_list(combined.counts) == [4] * 5
