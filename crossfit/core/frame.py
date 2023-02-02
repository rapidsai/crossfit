# from typing import Type

# from crossfit.core.metric import MetricState
# from crossfit.dataframe.core import CrossFrame


# class MetricFrame:
#     def __init__(self, state_df: CrossFrame, metric=None, data=None, index=None):
#         if not metric:
#             if "cls" not in state_df.attrs:
#                 raise ValueError("Please provide a `metric`")
#             metric = state_df.attrs["cls"]
#         self.metric = metric
#         self.state_df = state_df
#         self.data = data
#         self.index = index

#     @property
#     def state(self):
#         state_type: Type[MetricState] = self.metric.state_type()
#         state = state_type.from_state_df(self.state_df)

#         return state

#     @property
#     def group_names(self):
#         if self.data is None:
#             return None

#         group_cols = set(self.data.columns) - {"col"}
#         names = None
#         for c in list(self.data.columns):
#             if c not in group_cols:
#                 continue
#             part = f"{c}=" + self.data.select_column(c).astype(str)
#             if names is None:
#                 names = part
#             else:
#                 names = names + "&" + part

#         return names

#     def all(self):
#         return self.state_df.concat([self.state_df, self.data], axis=1)

#     def result(self, pivot=True):
#         metric_result = self.metric.present(self.state)
#         if isinstance(metric_result, MetricState):
#             metric_result = metric_result.state_dict
#         if not isinstance(metric_result, dict):
#             metric_result = {"result": metric_result}

#         result_df = self.state_df.from_dict(metric_result, index=self.index)

#         if self.data is not None:
#             df = self.state_df.concat(
#                 [self.data, self.state_df.from_dict(metric_result)], axis=1
#             )
#             if pivot:
#                 # Explicit column projection is work-around for conflicting
#                 # pandas/cudf behavior in older versions of cudf
#                 return (
#                     df.pivot(
#                         index=list(set(self.data.columns) - {"col"}), columns=["col"]
#                     )
#                     .project_columns(list(metric_result.keys()))
#                     .data
#                 )

#             return df.data

#         return result_df.data
