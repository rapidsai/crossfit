from typing import Type

import pandas as pd

from crossfit.core.metric import MetricState


class MetricFrame:
    def __init__(self, state_df, metric=None, data=None, index=None):
        if not metric:
            if "cls" not in state_df.attrs:
                raise ValueError("Please provide a `metric`")
            metric = state_df.attrs["cls"]
        self.metric = metric
        self.state_df = state_df
        self.data = data
        self.index = index

    @property
    def state(self):
        state_type: Type[MetricState] = self.metric.state_type()
        state = state_type.from_state_df(self.state_df)

        return state

    def all(self):
        return pd.concat([self.state_df, self.data], axis=1)

    def result(self):
        metric_result = self.metric.present(self.state)
        if isinstance(metric_result, MetricState):
            metric_result = metric_result.state_dict
        if not isinstance(metric_result, dict):
            metric_result = {"result": metric_result}

        result_df = pd.DataFrame(metric_result, index=self.index)

        if self.data is not None:
            df = pd.concat([self.data, pd.DataFrame(metric_result)], axis=1)
            df = df.pivot(index=set(self.data.columns) - set(["col"]), columns=["col"])

            return df

        return result_df
