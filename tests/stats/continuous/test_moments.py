import pytest

import numpy as np
import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.continuous.moments import Moments
from tests.utils import sample_df, to_list

data = {
    "a": list(range(5)) * 2,
    "a2": list(range(5)) * 2,
    "b": np.random.rand(10),
    "c": np.random.rand(10),
}


@sample_df(data)
def test_moments_per_col(df):
    mf: MetricFrame = calculate_per_col(Moments(), df)
    assert isinstance(mf, MetricFrame)
    assert list(mf.state_df.columns) == ["count", "mean", "var"]

    result = mf.result()
    assert isinstance(result, type(df))
    assert list(to_list(result.index)) == ["a", "a2", "b", "c"]
    np.testing.assert_allclose(to_list(result["mean"]), to_list(df.mean()))
    np.testing.assert_allclose(to_list(result["var"]), to_list(df.var()))
    np.testing.assert_allclose(to_list(result["std"]), to_list(df.std()))


@sample_df(data)
def test_moments_per_col_grouped(df):
    mf: MetricFrame = calculate_per_col(Moments(), df, groupby=["a", "a2"])
    assert isinstance(mf, MetricFrame)
    assert sorted(list(mf.data.columns)) == ["a", "a2", "col"]
    assert len(mf.data) == 10

    result = mf.result()
    assert isinstance(result, type(df))
    assert isinstance(result.columns, pd.MultiIndex)
    assert set(result.columns.levels[0]) == {"mean", "var", "count", "std"}
    assert sorted(list(result.columns.levels[1])) == ["b", "c"]
    assert sorted(list(result.index.names)) == ["a", "a2"]


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
@pytest.mark.parametrize("groupby", [None, "a"])
@pytest.mark.parametrize("npartitions", [1, 4])
def test_moments_dd(df, groupby, npartitions):
    dd = pytest.importorskip("dask.dataframe")

    from crossfit.dask.calculate import calculate_per_col as calculate_dask

    ddf = dd.from_pandas(df, npartitions=npartitions)
    metric = Moments()
    mf: MetricFrame = calculate_dask(metric, ddf, groupby=groupby)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    expect = calculate_per_col(Moments(), df, groupby=groupby).result()
    dd.assert_eq(result, expect)
