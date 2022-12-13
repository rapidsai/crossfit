from typing import overload

import pandas as pd

from crossfit.core.frame import MetricFrame
from crossfit.core.metric import Metric, MetricState, Array


@overload
def calculate(metric: Metric, data: pd.DataFrame, *args, **kwargs) -> MetricFrame:
    ...


@overload
def calculate(metric: Metric, data: Array, *args, **kwargs) -> MetricFrame:
    ...


@overload
def calculate(
    metric: Metric, data: Array, comparison: Array, *args, **kwargs
) -> MetricFrame:
    ...


def calculate(metric: Metric, data, *args, **kwargs) -> MetricFrame:
    state: MetricState = metric.prepare(data, *args, **kwargs)
    state_df = state.state_df()

    mf = MetricFrame(state_df, metric=metric)

    return mf


def calculate_per_col(metric: Metric, df, *args, **kwargs) -> MetricFrame:
    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        return _calculate_grouped_per_col(metric, df)

    rows = []
    index = []
    for name_col, col in df.items():
        state: MetricState = metric.prepare(col, *args, **kwargs)
        state_df = state.state_df()
        index.append(name_col)
        rows.append(state_df)

    df = pd.concat(rows, axis=0)
    mf = MetricFrame(df, metric=metric, index=pd.Index(index, name="col"))

    return mf


def _calculate_grouped_per_col(
    metric: Metric, df_grouped, *args, **kwargs
) -> MetricFrame:
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col, *args, **kwargs)
            state_df = state.state_df()
            if not isinstance(slice_key, tuple):
                slice_key = (slice_key,)
            index.append((name_col,) + slice_key)

            rows.append(state_df)
    keys = df_grouped.keys
    if isinstance(keys, str):
        keys = [keys]

    pd_index = pd.MultiIndex.from_tuples(index, names=["col"] + keys)
    df = pd.concat(rows, axis=0)
    df.index = pd_index

    df = df.reset_index()
    cols = df[pd_index.names]
    state_df = df[set(df.columns) - set(pd_index.names)]
    mdf = MetricFrame(state_df, metric=metric, data=cols)

    return mdf
