import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.moments import Moments

df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
    }
)


def test_moments_per_col():
    mdf: MetricFrame = calculate_per_col(df, Moments())
    assert isinstance(mdf, MetricFrame)
    assert list(mdf.state_df.columns) == ["count", "mean", "var"]

    result = mdf.result()
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["a", "a2", "b", "c"]


def test_moments_per_col_grouped():
    mdf: MetricFrame = calculate_per_col(df.groupby(["a", "a2"]), Moments())
    assert isinstance(mdf, MetricFrame)
    assert sorted(list(mdf.data.columns)) == ["a", "a2", "col"]
    assert len(mdf.data) == 10

    result = mdf.result()
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)
    assert list(result.columns.levels[0]) == ["mean", "variance"]
    assert sorted(list(result.columns.levels[1])) == ["b", "c"]
    assert sorted(list(result.index.names)) == ["a", "a2"]
