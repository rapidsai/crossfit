import dask.dataframe as dd

from crossfit.calculate.aggregate import Aggregator, metric_key
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.metrics.continuous.range import Range
from crossfit.metrics import Mean

from tests.utils import is_leaf_node_instance_of, sample_df


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation(df, npartitions=2):
    frame = CrossFrame(dd.from_pandas(df, npartitions=npartitions))

    # Test per-column behavior
    agg = Aggregator(Range(axis=0), per_column=True)
    test = frame.aggregate(agg)
    assert all(isinstance(x, Range) for x in test.values())

    # Test single-column behavior
    agg = Aggregator(Range(axis=0), post_group=lambda x: x["b"])
    test = frame.aggregate(agg)
    assert len(test) == 1
    assert isinstance(test[metric_key("Range")], Range)


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation_grouped(df, npartitions=2):
    frame = CrossFrame(dd.from_pandas(df, npartitions=npartitions))

    # Test per-column behavior
    agg = Aggregator(Range(axis=0), per_column=True, groupby=["a"])
    test = frame.aggregate(agg)
    assert is_leaf_node_instance_of(test, Range)

    # Test single-column behavior
    agg = Aggregator(Range(axis=0), pre=lambda x: x["b"], groupby=["a"])
    test = frame.aggregate(agg)
    assert is_leaf_node_instance_of(test, Range)

    # Simple to_frame=True case
    agg = Aggregator(
        {"mean": Mean(axis=0), "range": Range(axis=0)},
        per_column=True,
        post_group=lambda x: x[["b"]],
        groupby=["a"],
    )
    test = frame.aggregate(agg, to_frame=True)
    assert test.at[("a", 1, "b"), "range.min"] == 0

    # Complicated to_frame=True case
    agg1 = Aggregator(Range(axis=0), per_column=True, groupby=["a"])
    agg2 = Aggregator(Mean(axis=0), per_column=True, groupby=["a"])
    agg3 = Aggregator(Mean(axis=0), per_column=True, post_group=lambda x: x[["b"]])
    agg = Aggregator({"range": agg1, "mean": agg2, "mean-all": agg3})
    test = frame.aggregate(agg, to_frame=True)
    assert test.at[("a", 2, "b"), "Mean"] == 3000.0
    assert test.at[("a", 2, "b"), "Range.max"] == 4999.0
