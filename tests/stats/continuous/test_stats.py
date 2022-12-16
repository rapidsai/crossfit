import numpy as np

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.continuous.stats import ContinuousStats
from tests.utils import sample_df

data = {
    "a": list(range(5)) * 2,
    "a2": list(range(5)) * 2,
    "b": np.random.rand(10),
    "c": np.random.rand(10),
}


@sample_df(data)
def test_continuous_stats_per_col(df):
    mf: MetricFrame = calculate_per_col(ContinuousStats(), df)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))
    assert set(result.columns) == {
        "common.num_missing",
        "common.count",
        "range.min",
        "range.max",
        "moments.mean",
        "moments.std",
        "moments.var",
        "moments.count",
    }


@sample_df(data)
def test_continuous_stats_per_col_grouped(df):
    mf: MetricFrame = calculate_per_col(ContinuousStats(), df.groupby(["a", "a2"]))
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))
    assert len(result.index) == 5
    assert len(result.columns) == 16
