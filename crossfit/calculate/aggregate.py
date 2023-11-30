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

import types
from collections import defaultdict, namedtuple
from functools import wraps

import numpy as np

from crossfit.data.array.conversion import convert_array
from crossfit.data.dataframe.core import FrameBackend


def pre_processing(func):
    @wraps(func)
    def wrapper(self, data, *args, **kwargs):
        if self.pre:
            data = self.pre(data, *args, **kwargs)

            # # TODO: This is a hack, fix this
            if isinstance(data, tuple):
                return func(*data, **kwargs)

        return func(data, *args, **kwargs)

    return wrapper


MetricKey = namedtuple("MetricKey", "grouping group column name")


def metric_key(name, column=None, grouping=None, group=None):
    return MetricKey(grouping, group, column, name)


class Aggregator:
    def __init__(
        self,
        aggs=None,
        pre=None,
        post_group=None,
        post=None,
        groupby=None,
        per_column=False,
        axis=0,
        name=None,
    ):
        if aggs and not isinstance(aggs, dict):
            name = name if name is not None else type(aggs).__name__
            aggs = {name: aggs}
        self.aggs = aggs
        self.pre = pre
        self.post_group = post_group
        self.post = post
        if isinstance(groupby, (str, int, tuple)):
            groupby = [groupby]
        self.groupby = groupby
        self.per_column = per_column
        self.axis = axis

    def _prepare(self, data, *args, **kwargs):
        if self.aggs:
            return {name: agg(data, *args, **kwargs) for name, agg in self.aggs.items()}
        return data

    def prepare(self, data, *args, **kwargs):
        if isinstance(data, FrameBackend):
            return self._prepare_frame(data, **kwargs)
        return self._prepare(data, *args, **kwargs)

    def _prepare_frame(self, data, **kwargs):
        if not isinstance(data, FrameBackend):
            raise ValueError()
        if not self.aggs:
            raise NotImplementedError()
        state = {}
        groups = data.groupby_partition(self.groupby) if self.groupby else {None: data}
        for slice_key, group_df in groups.items():
            grouping = None
            if self.groupby:
                if not isinstance(slice_key, tuple):
                    slice_key = (slice_key,)
                grouping = tuple(self.groupby)
            if self.post_group:
                group_df = self.post_group(group_df)

            if self.per_column:
                # TODO: Not sure if this is the best way to do this
                if callable(self.per_column):
                    columns = self.per_column(group_df)
                else:
                    columns = group_df.columns
            else:
                columns = [None]

            for column in columns:
                group_df_col = group_df
                if isinstance(group_df, FrameBackend) and column is not None:
                    group_df_col = group_df[column]
                if not isinstance(group_df, list):
                    group_df_col = [group_df_col]
                for name, result in self._prepare(
                    *group_df_col,
                    **kwargs,
                ).items():
                    state[
                        metric_key(
                            name,
                            grouping=grouping,
                            group=slice_key,
                            column=column,
                        )
                    ] = result
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

    def present(self, state, to_frame=True):
        if to_frame:
            # TODO: Clean this up and generalize from pandas
            import pandas as pd

            new = defaultdict(dict)
            present_dict = self.present(state, to_frame=False)
            keys = list(present_dict.keys())

            if isinstance(keys[0], str):
                # return pd.DataFrame(present_dict)
                result = {k: convert_array(v, np.ndarray) for k, v in present_dict.items()}

                return pd.DataFrame.from_dict(result, orient="index").T

            groupings = {"&".join(k.grouping) if k.grouping else None for k in keys}
            columns = {k.column for k in keys}

            if columns and groupings != {None}:
                for k, v in present_dict.items():
                    grouping = "&".join(k.grouping) if isinstance(k.grouping, tuple) else k.grouping
                    if isinstance(k.group, tuple):
                        if len(k.group) > 1:
                            group = "&".join([str(i) for i in k.group])
                        else:
                            group = k.group[0]
                    else:
                        group = k.group
                    if isinstance(v, dict):
                        new[(grouping, group, k.column)].update(
                            {
                                ((k.name + "." + _k) if _k != k.name else k.name): _v
                                for _k, _v in v.items()
                            }
                        )
                    else:
                        new[(grouping, group, k.column)].update({k.name: v})
                index = pd.MultiIndex.from_tuples(new.keys(), names=("grouping", "group", "column"))
                output = pd.DataFrame.from_records(list(new.values()), index=index)

                if columns == {None}:
                    output.index = output.index.droplevel("column")

                return output

            elif columns:
                new = defaultdict(dict)
                for k, v in present_dict.items():
                    new[k.column].update({k.name: v})
                data = [{"column": key, **val} for key, val in new.items()]

                return pd.DataFrame(data).set_index("column")
            else:
                new = defaultdict(dict)
                # TODO: Handle multiple groupings
                grouping = groupings[0]

                for k, v in present_dict.items():
                    if len(v) == 1:
                        v = v[0]
                    new["&".join(k.group)].update({k.name: v})

                data = [{grouping: key, **val} for key, val in new.items()]
                return pd.DataFrame(data).set_index(grouping)
        if isinstance(state, dict):
            return present_state_dict(state)
        return state

    def __call__(
        self,
        data,
        *args,
        **kwargs,
    ):
        return self.prepare(data, *args, **kwargs)

    def join(self, *other: "Aggregator"):
        ...


def present_state_dict(state, key=None):
    result = {}
    for k in state.keys():
        if isinstance(k, MetricKey) or key is None:
            _k = k
        else:
            # TODO: Why is this needed?
            if not isinstance(key, MetricKey) and isinstance(key, tuple):
                key = MetricKey(*key)
            assert isinstance(key, MetricKey)
            assert isinstance(k, str)

            if key.name:
                name = key.name + "." + k
            else:
                name = k

            _k = metric_key(
                name,
                column=key.column,
                grouping=key.grouping,
                group=key.group,
            )

        if hasattr(state[k], "present"):
            metric_result = state[k].present()
            if isinstance(metric_result, dict):
                result.update(present_state_dict(metric_result, key=_k))
            else:
                result[_k] = metric_result
        elif isinstance(state[k], dict):
            result.update(present_state_dict(state[k], key=_k))
        else:
            result[_k] = state[k]

    return result


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
