from collections import defaultdict

from crossfit.core.frame import MetricFrame
from crossfit.core.metric import Metric
from crossfit.dataframe.dispatch import frame_dispatch

from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
import dask.dataframe as dd


def _map_reduce(ddf, metric, map_func, reduce_func, finalize_func):

    bag = ddf.to_bag(format="frame")
    if ddf.npartitions == 1:
        reduced = bag.map_partitions(map_func)
    else:
        reduced = bag.reduction(
            map_func,
            reduce_func,
            split_every=32,
            out_type=metric.state_type(),
            name="reduced",
        )
    final = reduced.map_partitions(finalize_func)
    name = f"result-{final.name}"
    graph = HighLevelGraph.from_collections(
        name, {name: (final.name, 0)}, dependencies=[final]
    )
    return Delayed(name, graph).compute()


def calculate_per_col(metric: Metric, ddf: dd.DataFrame, groupby=None) -> MetricFrame:

    # If groupby is specified, use special code path
    if groupby:
        return _calculate_grouped_per_col(metric, ddf, groupby)

    ref_adf = frame_dispatch(ddf._meta)

    def _map(df) -> dict:
        adf = frame_dispatch(df)
        return {col: metric.prepare(adf.select_column(col)) for col in adf.columns}

    def _reduce(partitions) -> dict:
        col_states = defaultdict(list)
        for partition in partitions:
            for col, state in partition.items():
                col_states[col].append(state)
        states = {}
        for col, state_list in col_states.items():
            states[col] = state_list[0].concat(*state_list[1:]).reduce()
        return states

    def _finalize(column_states: dict) -> MetricFrame:
        # Loop over columns
        rows = []
        index = []
        for name_col, state in column_states.items():
            state_df = state.state_df(ref_adf)
            index.append(name_col)
            rows.append(state_df)

        # Return a MetricFrame summary
        return MetricFrame(
            ref_adf.concat(rows, axis=0, ignore_index=True),
            metric=metric,
            index=ref_adf.from_dict({"col": index}).select_column("col"),
        )

    return _map_reduce(ddf, metric, _map, _reduce, _finalize)


def _calculate_grouped_per_col(
    metric: Metric, ddf: dd.DataFrame, keys: list
) -> MetricFrame:

    ref_adf = frame_dispatch(ddf._meta)
    keys = [keys] if isinstance(keys, (str, int)) else list(keys)

    def _map(df) -> dict:
        adf = frame_dispatch(df)
        groups = adf.groupby_partition(keys)
        states = {}
        for slice_key, group_df in groups.items():
            if not isinstance(slice_key, tuple):
                slice_key = (slice_key,)
            states[slice_key] = {}
            for name_col in group_df.columns:
                if name_col in keys:
                    continue
                state = metric.prepare(group_df.select_column(name_col))
                states[slice_key][name_col] = state
        return states

    def _reduce(partitions) -> dict:
        all_states = {}
        for partition in partitions:
            for slice_key, slice_states in partition.items():
                if slice_key not in all_states:
                    all_states[slice_key] = defaultdict(list)
                for col_name, state in slice_states.items():
                    all_states[slice_key][col_name].append(state)
        states = {}
        for slice_key, column_states in all_states.items():
            states[slice_key] = {}
            for col_name, state_list in column_states.items():
                states[slice_key][col_name] = (
                    state_list[0].concat(*state_list[1:]).reduce()
                )
        return states

    def _finalize(slice_states: dict) -> MetricFrame:

        # Loop over columns
        rows = []
        index = []
        cols = defaultdict(list)
        for slice_key, column_states in slice_states.items():
            for name_col, state in column_states.items():
                if name_col in keys:
                    continue
                state_df = state.state_df(ref_adf)
                index.append(name_col)
                rows.append(state_df)
                cols["col"].append(name_col)
                for i, k in enumerate(keys):
                    cols[k].append(slice_key[i])

        # Return a MetricFrame summary
        return MetricFrame(
            ref_adf.concat(rows, axis=0, ignore_index=True),
            metric=metric,
            data=ref_adf.from_dict(cols),
        )

    return _map_reduce(ddf, metric, _map, _reduce, _finalize)
