import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.continuous.moments import Moments

df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
    }
)


def test_moments_per_col():
    mf: MetricFrame = calculate_per_col(df, Moments())
    assert isinstance(mf, MetricFrame)
    assert list(mf.state_df.columns) == ["count", "mean", "var"]

    result = mf.result()
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["a", "a2", "b", "c"]
    np.testing.assert_allclose(result["mean"], df.mean())
    np.testing.assert_allclose(result["std"], df.std())
    np.testing.assert_allclose(result["var"], df.var())


def test_moments_per_col_grouped():
    mf: MetricFrame = calculate_per_col(df.groupby(["a", "a2"]), Moments())
    assert isinstance(mf, MetricFrame)
    assert sorted(list(mf.data.columns)) == ["a", "a2", "col"]
    assert len(mf.data) == 10

    result = mf.result()
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.columns, pd.MultiIndex)
    assert set(result.columns.levels[0]) == {"mean", "var", "count", "std"}
    assert sorted(list(result.columns.levels[1])) == ["b", "c"]
    assert sorted(list(result.index.names)) == ["a", "a2"]
