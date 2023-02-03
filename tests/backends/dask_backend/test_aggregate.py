import dask.dataframe as dd

from crossfit.backends.dask.aggregate import aggregate
from tests.utils import sample_df
from crossfit.metrics.continuous.range import Range


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)
    test = aggregate(ddf, Range(axis=0), per_col=True)

    assert all(isinstance(x, Range) for x in test.values())
