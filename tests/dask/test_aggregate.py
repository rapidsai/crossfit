import pytest

import crossfit as cf
from crossfit.dask.aggregate import aggregate
from tests.utils import sample_df
from crossfit.stats.continuous.common import RangeState


class SomeAgg(cf.Aggregator):
    def prepare(self, data):
        return RangeState(data.min(axis=0), data.max(axis=0))


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_dask_aggregation(df, npartitions=2):
    dd = pytest.importorskip("dask.dataframe")

    ddf = dd.from_pandas(df, npartitions=npartitions)
    test = aggregate(ddf, SomeAgg(), per_col=True)

    assert all(isinstance(x, RangeState) for x in test.values())
