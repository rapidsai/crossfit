# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
