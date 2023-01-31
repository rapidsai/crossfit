import pytest

import numpy as np

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.continuous.stats import ContinuousStats
from tests.utils import sample_df, to_list

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
    mf: MetricFrame = calculate_per_col(ContinuousStats(), df, groupby=["a", "a2"])
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))
    assert len(result.index) == 5
    assert len(result.columns) == 16


@sample_df({"b": range(10, 50)})
def test_continuous_stats_reduce(df):
    # Use simple DataFrame
    stride = 10
    ser = df["b"]
    size_ser = len(ser)
    sers = [ser.iloc[i : i + stride] for i in range(0, size_ser, stride)]

    # Prepare, concat, and reduce states
    metric = ContinuousStats()
    states = [metric.prepare(s) for s in sers]
    concatenated = states[0].concat(*states[1:])
    reduced = concatenated.reduce()

    # Check reduced-state result
    assert to_list(reduced.range.min) == [ser.min()]
    assert to_list(reduced.range.max) == [ser.max()]
    assert to_list(reduced.moments.count) == [len(ser)]
    assert to_list(reduced.common.count) == [len(ser)]
    assert to_list(reduced.common.num_missing) == [0]
    np.testing.assert_allclose(to_list(reduced.moments.mean), [ser.mean()])
    np.testing.assert_allclose(to_list(reduced.moments.var), [ser.var()])


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
@pytest.mark.parametrize("groupby", [None, "a"])
@pytest.mark.parametrize("npartitions", [1, 4])
def test_continuous_stats_dd(df, groupby, npartitions):
    dd = pytest.importorskip("dask.dataframe")

    from crossfit.dask.calculate import calculate_per_col as calculate_dask

    ddf = dd.from_pandas(df, npartitions=npartitions)
    metric = ContinuousStats()
    mf: MetricFrame = calculate_dask(metric, ddf, groupby=groupby)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    expect = calculate_per_col(ContinuousStats(), df, groupby=groupby).result()
    dd.assert_eq(result, expect)
