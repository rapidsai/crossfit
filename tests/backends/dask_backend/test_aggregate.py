import dask.dataframe as dd

from crossfit.backends.dask.aggregate import aggregate
from crossfit.metrics.continuous.range import Range

from tests.utils import is_leaf_node_instance_of, sample_df


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)
    test = aggregate(ddf, Range(axis=0), per_col=True)

    assert all(isinstance(x, Range) for x in test.values())


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation_grouped(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)
    test = aggregate(ddf, Range(axis=0), groupby=["a"], per_col=True)

    assert is_leaf_node_instance_of(test, Range)
