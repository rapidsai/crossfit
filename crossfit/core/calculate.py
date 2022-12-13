import pandas as pd

from crossfit.core.frame import MetricFrame
from crossfit.core.metric import Metric, MetricState


def _calculate_grouped_per_col(df_grouped, metric: Metric) -> MetricFrame:
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col)
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


def calculate_per_col(df, metric: Metric) -> MetricFrame:
    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        return _calculate_grouped_per_col(df, metric)

    rows = []
    index = []
    for name_col, col in df.items():
        state: MetricState = metric.prepare(col)
        state_df = state.state_df()
        index.append(name_col)
        rows.append(state_df)

    df = pd.concat(rows, axis=0)
    mf = MetricFrame(df, metric=metric, index=pd.Index(index, name="col"))

    return mf
