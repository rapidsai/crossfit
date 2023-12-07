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

# from typing import overload
# from collections import defaultdict

# import pandas as pd

# from crossfit.core.frame import MetricFrame
# from crossfit.core.metric import Metric, MetricState, Array
# from crossfit.dataframe.core import CrossFrame
# from crossfit.dataframe.dispatch import frame_dispatch


# @overload
# def calculate(metric: Metric, data: pd.DataFrame, *args, **kwargs) -> MetricFrame:
#     ...


# @overload
# def calculate(metric: Metric, data: Array, *args, **kwargs) -> MetricFrame:
#     ...


# @overload
# def calculate(
#     metric: Metric, data: Array, comparison: Array, *args, **kwargs
# ) -> MetricFrame:
#     ...


# def calculate(metric: Metric, data, *args, **kwargs) -> MetricFrame:
#     state: MetricState = metric.prepare(data, *args, **kwargs)
#     try:
#         adf = frame_dispatch(data)
#     except TypeError:
#         # Fall-back to Pandas when the array type
#         # has no registered AbstractFrame class
#         adf = frame_dispatch(pd.DataFrame())
#     state_df = state.state_df(adf)

#     mf = MetricFrame(state_df, metric=metric)

#     return mf


# def calculate_per_col(metric: Metric, df, *args, groupby=None, **kwargs) -> MetricFrame:

#     # Wrape df in an AbstractFrame class
#     adf = frame_dispatch(df)

#     # If groupby is specified, use special code path
#     if groupby:
#         return _calculate_grouped_per_col(
#             metric,
#             adf,
#             groupby,
#             *args,
#             **kwargs,
#         )

#     # Loop over columns
#     rows = []
#     index = []
#     for name_col in adf.columns:
#         col = adf.select_column(name_col)
#         state: MetricState = metric.prepare(col, *args, **kwargs)
#         state_df = state.state_df(adf)
#         index.append(name_col)
#         rows.append(state_df)

#     # Return a MetricFrame summary
#     return MetricFrame(
#         adf.concat(rows, axis=0, ignore_index=True),
#         metric=metric,
#         index=adf.from_dict({"col": index}).select_column("col"),
#     )


# def _calculate_grouped_per_col(
#     metric: Metric, adf: CrossFrame, keys: list, *args, **kwargs
# ) -> MetricFrame:

#     # Perform groupby partitioning
#     keys = [keys] if isinstance(keys, (str, int)) else list(keys)
#     groups = adf.groupby_partition(keys)

#     # Nested loop over groups and columns
#     rows = []
#     cols = defaultdict(list)
#     for slice_key, group_df in groups.items():
#         for name_col in group_df.columns:
#             if name_col in keys:
#                 continue
#             col = group_df.select_column(name_col)
#             state = metric.prepare(col, *args, **kwargs)
#             state_df = state.state_df(group_df)
#             if not isinstance(slice_key, tuple):
#                 slice_key = (slice_key,)
#             rows.append(state_df)
#             cols["col"].append(name_col)
#             for i, k in enumerate(keys):
#                 cols[k].append(slice_key[i])

#     # Return a MetricFrame summary
#     return MetricFrame(
#         adf.concat(rows, axis=0, ignore_index=True),
#         metric=metric,
#         data=adf.from_dict(cols),
#     )
