from functools import partial

import dask.dataframe as dd
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph

from crossfit.calculate.aggregate import Aggregator
from crossfit.data.dataframe.dispatch import CrossFrame


def aggregate(
    ddf: dd.DataFrame,
    aggregator: Aggregator,
    to_frame=False,
    map_kwargs=None,
    compute_kwargs=None,
):
    if not isinstance(aggregator, Aggregator):
        raise TypeError(f"Expected `Aggregator`, got {type(aggregator)}")
    map_func = partial(aggregator, **(map_kwargs or {}))

    def reduce_func(vals):
        vals = list(vals)
        return aggregator.reduce(*vals)

    bag = ddf.to_bag(format="frame").map_partitions(CrossFrame)
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

    result = Delayed(name, graph).compute(**(compute_kwargs or {}))
    if to_frame:
        return aggregator.present(result, to_frame=True)

    return result
