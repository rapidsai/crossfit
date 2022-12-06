import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.continuous.stats import ContinuousStats

df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
    }
)


def test_continuous_stats_per_col():
    mf: MetricFrame = calculate_per_col(df, ContinuousStats())
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "common.num_missing",
        "common.count",
        "range.min",
        "range.max",
        "moments.mean",
        "moments.var",
        "moments.count",
    }


def test_continuous_stats_per_col_grouped():
    mf: MetricFrame = calculate_per_col(df.groupby(["a", "a2"]), ContinuousStats())
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == 5
    assert len(result.columns) == 14
