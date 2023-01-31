import pytest

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.categorical.stats import CategoricalStats
from tests.utils import sample_df, to_list

data = {
    "a": ["a", "b", "c", "d", "e"] * 2,
    "a2": ["f", "g", "h", "i", "j"] * 2,
    "b": ["foo", "bar"] * 5,
    "c": ["hello", "world"] * 5,
}


@sample_df(data)
def test_categorical_stats_per_col(df):
    mf: MetricFrame = calculate_per_col(CategoricalStats(), df)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))


@sample_df(data)
def test_categorical_stats_reduce(df):
    # Use single columns of df
    stride = 3
    ser = df["b"]
    size_ser = len(ser)
    sers = [ser.iloc[i : i + stride] for i in range(0, size_ser, stride)]

    # Prepare, concat, and reduce states
    metric = CategoricalStats()
    states = [metric.prepare(s) for s in sers]
    concatenated = states[0].concat(*states[1:])
    reduced = concatenated.reduce()

    # Check reduced-state result
    expected = metric.prepare(ser)
    assert to_list(reduced.lengths.count) == [to_list(expected.lengths.count)]
    assert to_list(reduced.lengths.sum) == [to_list(expected.lengths.sum)]
    assert to_list(reduced.common.count) == [to_list(expected.common.count)]
    assert to_list(reduced.common.num_missing) == [to_list(expected.common.num_missing)]
    assert to_list(reduced.value_counts.values) == to_list(expected.value_counts.values)
    assert to_list(reduced.value_counts.counts) == to_list(expected.value_counts.counts)


@sample_df(data)
def test_categorical_stats_dd(df):
    dd = pytest.importorskip("dask.dataframe")

    from crossfit.dask.calculate import calculate_per_col as calculate_dask

    ddf = dd.from_pandas(df, npartitions=2)

    mf: MetricFrame = calculate_dask(CategoricalStats(), ddf)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))
    expect = calculate_per_col(CategoricalStats(), df).result()
    if hasattr(result["top_values"], "list"):
        # Need to sort list values to compare cudf results
        # TODO: Fix this
        result["top_values"] = result["top_values"].list.sort_values()
        expect["top_values"] = expect["top_values"].list.sort_values()
    dd.assert_eq(result, expect)
