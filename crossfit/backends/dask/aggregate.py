from functools import partial
from typing import Sequence, Union

from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
import dask.dataframe as dd

from crossfit.calculate.aggregate import Aggregator
from crossfit.metrics.base import CrossMetric


def aggregate(
    ddf: dd.DataFrame,
    to_aggregate: Union[Aggregator, CrossMetric],
    groupby: Sequence[str] = (),
    per_col=False,
):
    if isinstance(to_aggregate, CrossMetric):
        aggregator = to_aggregate.to_aggregator()
    else:
        aggregator = to_aggregate
    map_func = partial(aggregator, groupby=groupby, per_col=per_col)

    def reduce_func(vals):
        vals = list(vals)
        return aggregator.reduce(*vals)

    bag = ddf.to_bag(format="frame")
    if ddf.npartitions == 1:
        reduced = bag.map_partitions(map_func)
        name = f"result-{reduced.name}"
        graph = HighLevelGraph.from_collections(
            name, {name: (reduced.name, 0)}, dependencies=[reduced]
        )
    else:
        reduced = bag.reduction(
            map_func,
            reduce_func,
            split_every=32,
            name="reduced",
        )
        name = reduced.name
        graph = reduced.dask

    return Delayed(name, graph).compute()