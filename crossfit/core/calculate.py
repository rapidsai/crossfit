import pandas as pd

from crossfit.core.metric import Metric
from crossfit.core.frame import MetricFrame


def _calculate_grouped_per_col(df_grouped, metric: Metric) -> MetricFrame:
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col)
            state_df = state.state_df()
            index.append((name_col,) + slice_key)
            
            rows.append(state_df)
    if len(df_grouped.keys) > 1:
        pd_index = pd.MultiIndex.from_tuples(index, names=["col"] + df_grouped.keys)
    else:
        pd_index = pd.Index(index, name=df_grouped.keys)
        
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
        state = metric.prepare(col)
        state_df = state.state_df()
        index.append(name_col)
        rows.append(state_df)
        
    df = pd.concat(rows, axis=0)    
    mdf = MetricFrame(df, metric=metric, index=pd.Index(index, name="col"))
    
    return mdf
