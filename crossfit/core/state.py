import abc
from dataclasses import fields
from typing import Generic, Optional, TypeVar

import numpy as np
import pandas as pd

from crossfit.dispatch.array import ToType, convert

ArrayType = TypeVar("ArrayType", np.ndarray, pd.Series)


class MetricState(Generic[ArrayType], abc.ABC):
    @abc.abstractmethod
    def merge(self, other):
        raise NotImplementedError()

    def __post_init__(self):
        for field in fields(self):
            val = getattr(self, field.name)
            if field.type == ArrayType:
                if isinstance(val, (int, float, str, list, tuple)):
                    object.__setattr__(self, field.name, self.to_state(val))

    def to_state(self, val) -> ArrayType:
        return np.array(val)

    def __add__(self, other):
        return self.merge(other)

    def convert(self, type: ToType) -> "MetricState[ToType]":
        params = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, (int, float, str)):
                params[field.name] = val
            elif isinstance(val, MetricState):
                params[field.name] = val.convert(type)
            else:
                params[field.name] = convert(val, type)

        return self.__class__(**params)

    def concat(self, *other, axis=None):
        params = {}
        for field in fields(self):
            if field.type == ArrayType:
                params[field.name] = np.concatenate(
                    [
                        getattr(self, field.name),
                        *map(lambda x: getattr(x, field.name), other),
                    ],
                    axis=axis,
                )

        return self.__class__(**params)

    @property
    def state_dict(self):
        output = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if field.type == ArrayType:
                output[field.name] = val
            elif isinstance(getattr(self, field.name), MetricState):
                for key, child_val in getattr(self, field.name).__state__.items():
                    output[".".join([field.name, key])] = child_val

        return output

    @classmethod
    def from_state(cls, state: dict):
        params = {}

        _state = unflatten_state(state)

        for field in fields(cls):
            if field.type == ArrayType:
                params[field.name] = _state[field.name]
            elif isinstance(field.type, type) and issubclass(field.type, MetricState):
                params[field.name] = field.type.from_state(_state[field.name])

        return cls(**params)

    def state_df(self, index: Optional[str] = None, **kwargs):
        if not index:
            index = [
                self.cls_path(),
            ]
        return pd.DataFrame(self.state_dict, index=list(index), **kwargs)

    @classmethod
    def from_state_df(cls, df: pd.DataFrame):
        d = {key: val.values for key, val in df.to_dict(orient="series").items()}
        return cls.from_state(d)

    @classmethod
    def cls_path(cls):
        return ".".join([cls.__module__, cls.__name__])


def nested_set_dict(d, keys, value):
    assert keys
    key = keys[0]
    if len(keys) == 1:
        if key in d:
            raise ValueError("duplicated key '{}'".format(key))
        d[key] = value
        return
    d = d.setdefault(key, {})
    nested_set_dict(d, keys[1:], value)


def unflatten_state(state):
    if not any("." in key for key in state.keys()):
        return state

    output = {}
    for flat_key, value in state.items():
        key_tuple = tuple(flat_key.split("."))
        nested_set_dict(output, key_tuple, value)

    return output
