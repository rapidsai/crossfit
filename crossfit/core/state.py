from typing import Union, Tuple, List, TypeVar, Type

import pandas as pd
import numpy as np

from crossfit.core.metric import Metric, MetricState


MetricStateT = TypeVar("MetricStateT", bound=MetricState)


class State:
    def __init__(self, state=None, data=None, index=None):
        self.data = data
        self.index = index

        if state is not None:
            if isinstance(state, dict):
                states = {}
                state_keys = {}
                for key, val in state.items():
                    state = val.state_df()
                    if not type(val) in states:
                        states[type(val)] = [state]
                        state_keys[type(val)] = [key]
                    else:
                        states[type(val)].append(state)
                        state_keys[type(val)].append(key)

                self.states = {}
                for key, val in states.items():
                    self.states[key] = pd.concat(val, axis=0)
                    self.states[key].index = pd.Index(state_keys[key])
        else:
            self.states = state

    def obj(self, obj_type: Type[MetricStateT] = None) -> MetricStateT:
        if obj_type == None:
            if len(self.states) == 1:
                obj_type = list(self.states.keys())[0]
            else:
                raise ValueError("Please provide an `obj_type`")

        return obj_type.from_state_df(self.states[obj_type])

    def __getattr__(self, key):
        for state_type in self.states.keys():
            try:
                return getattr(self.obj(state_type), key)
            except KeyError:
                pass

        raise AttributeError(key)

    def __repr__(self):
        if len(self.states) == 1:
            return list(self.states.values())[0].__repr__()
        else:
            return "State"

    def _repr_html_(self):
        # Repr for Jupyter Notebook
        return self.state._repr_html_()

    def combine(self, other: "State") -> "State":
        ...

    def join(
        self, other: Union["State", MetricState], on: Union[str, Tuple[str], List[str]]
    ) -> "State":
        ...

    def __add__(self, other) -> "State":
        return self.combine(other)


class PerColumn(Metric):
    def __init__(self, metric):
        self.metric = metric

    def prepare(self, data, *args, **kwargs):
        return State(
            {n: self.metric.prepare(col, *args, **kwargs) for n, col in data.items()}
        )


class Stats(Metric):
    def prepare(self, data):
        from crossfit.stats.categorical.stats import CategoricalStats
        from crossfit.stats.continuous.stats import ContinuousStats

        state = {}
        for n, col in data.items():
            if self.is_con(col):
                state[n] = ContinuousStats().prepare(col)
            elif self.is_cat(col):
                state[n] = CategoricalStats().prepare(col)

        return State(state=state)

    def is_con(self, col):
        return np.issubdtype(col.dtype, np.number)

    def is_cat(self, col):
        return col.dtype == "object"


class Grouped(Metric):
    def __init__(self, metric, by):
        self.metric = metric
        self.by = by

    def prepare(self, data):
        grouped = data.groupby(self.by)

        state = State()
        for name, slice in dict(grouped.groups).items():
            group = grouped.obj.iloc[slice]
            m = self.metric.prepare(group)
            state = state.join(m, name)

        return state
