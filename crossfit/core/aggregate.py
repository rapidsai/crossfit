import types
from typing import Sequence
from functools import reduce, wraps, partial

from crossfit.core.dataframe.dispatch import frame_dispatch


def pre_processing(func):
    @wraps(func)
    def wrapper(self, data):
        if self.pre:
            data = self.pre(data)

        return func(data)

    return wrapper


class Aggregator:
    def __init__(self, *aggs, pre=None, post=None):
        if aggs:
            if all(isinstance(agg, dict) for agg in aggs):
                aggs = reduce(lambda a, b: dict(a, **b), aggs)
            else:
                aggs = {agg.__name__: agg for agg in aggs}
        self.aggs = aggs
        self.pre = pre
        self.post = post

    def prepare(self, data):
        if self.aggs:
            if len(self.aggs) == 1:
                return next(iter(self.aggs.values()))(data)
            return {name: agg(data) for name, agg in self.aggs.items()}

        return data

    def call_per_col(self, data, to_call):
        return {col: to_call(data.select_column(col)) for col in data.columns}

    def call_grouped(self, data, groupby, to_call):
        state = {}
        groups = data.groupby_partition(groupby)
        for slice_key, group_df in groups.items():
            if not isinstance(slice_key, tuple):
                slice_key = (slice_key,)
            state[slice_key] = to_call(group_df)

        return state

    def __getattribute__(self, name: str):
        attr = object.__getattribute__(self, name)
        if name == "prepare" and self.pre is not None:
            prepare = pre_processing(attr)
            return types.MethodType(prepare, self)

        return attr

    def reduce(self, *values):
        if not values:
            raise ValueError("No values to reduce")

        if len(values) == 1:
            return values[0]

        reduced = values[0]
        for val in values[1:]:
            if isinstance(reduced, dict):
                reduced = reduce_state_dicts(reduced, val)
            else:
                reduced = reduced.combine(val)

        return reduced

    def present(self, state):
        return state

    def __call__(
        self, data, *args, groupby: Sequence[str] = (), per_col=False, **kwargs
    ):
        # TODO: Remove this in favor of detecting the type of data
        try:
            data = frame_dispatch(data)
        except TypeError:
            pass
        prepare = partial(self.prepare, *args, **kwargs)

        if per_col:
            prepare = partial(self.call_per_col, to_call=prepare)

        if groupby:
            return self.call_grouped(data, groupby, prepare)

        return prepare(data)

    def join(self, *other: "Aggregator"):
        ...


def reduce_state_dicts(dict1, dict2):
    result = {}
    for key in set(dict1.keys()).union(dict2.keys()):
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result[key] = reduce_state_dicts(dict1[key], dict2[key])
            else:
                result[key] = dict1[key].combine(dict2[key])
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]

    return result
