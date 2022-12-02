import abc
import inspect
from dataclasses import fields
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
from typing_utils import get_args, get_origin

from crossfit.dispatch.array import ToType, convert
from crossfit.utils.types import get_generic_type_arg

OutputType = TypeVar("OutputType")
ArrayType = TypeVar("ArrayType", np.ndarray, pd.Series)


@runtime_checkable
class Combiner(Protocol):
    def combiner(self, left, right):
        ...


StateCombiner = TypeVar("StateCombiner", bound=Combiner)


class State(Generic[StateCombiner]):
    @classmethod
    def generic_state_type(cls):
        generic = get_generic_type_arg(cls)

        if not generic:
            return None

        if issubclass(generic, Metric):
            return generic.state_type()

        return generic

    @classmethod
    def is_tensor(cls):
        generic = get_generic_type_arg(cls)
        if generic == ArrayType:
            return True

        return get_generic_type_arg(cls) is None

    @classmethod
    def is_child(cls):
        generic = get_generic_type_arg(cls)

        if not generic or generic == ArrayType:
            return False

        if issubclass(generic, (Metric, MetricState)):
            return True

        return False


class MetricState(Generic[ArrayType], abc.ABC):
    def combine(self, other):
        merged_fields = {}

        for field in fields(self):
            _type = get_origin(field.type)
            _type = _type or field.type
            if not _type == State:
                raise TypeError("Cannot add non-state field")

            _type_args = get_args(field.type)
            combiner = _type_args[0] if _type_args else None
            if combiner and hasattr(combiner, "combiner"):
                merged_fields[field.name] = combiner.combiner(
                    getattr(self, field.name),
                    getattr(other, field.name),
                )
            else:
                raise ValueError(
                    f"Cannot combine {self.__class__.__name__} automatically "
                    + "please implement a combiner function: `def combine(self, other)`"
                )

        return self.__class__(**merged_fields)

    def __post_init__(self):
        for field in fields(self):
            val = getattr(self, field.name)
            if field.type == ArrayType or (
                field.type == State and field.type.is_tensor()
            ):
                if isinstance(val, (int, float, str, list, tuple)):
                    object.__setattr__(self, field.name, self.to_state(val))

    def to_state(self, val) -> ArrayType:
        return np.array(val)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

        return self.combine(other)

    def combiner(self, left, right):
        return left.combine(right)

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

    @classmethod
    def field_is_tensor(cls, field):
        return field.type == ArrayType or (
            field.type == State and field.type.is_tensor()
        )

    @classmethod
    def field_is_child(cls, field):
        return isinstance(field.type, type) and (
            issubclass(field.type, MetricState)
            or (field.type == State and field.type.is_child())
        )


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


StateType = TypeVar("StateType", bound=MetricState, covariant=True)


class Metric(Generic[StateType], abc.ABC):
    @abc.abstractmethod
    def prepare(self, data: ArrayType, *args, **kwargs) -> StateType:
        raise NotImplementedError()

    def present(self, state: StateType) -> OutputType:
        return state

    @classmethod
    def state_type(cls):
        prepare_type = inspect.signature(cls.prepare).return_annotation
        if prepare_type:
            return prepare_type
        cls_type = get_generic_type_arg(cls)
        if cls_type:
            return cls_type

        raise ValueError("Could not infer state type for {}".format(cls))


class ComparisonMetric(Metric, abc.ABC):
    @abc.abstractmethod
    def prepare(self, data: ArrayType, comparison: ArrayType, **kwargs) -> StateType:
        raise NotImplementedError()

    def __call__(self, data: ArrayType, comparison: ArrayType, **kwargs) -> float:
        state = self.prepare(data, comparison, **kwargs)

        return self.__class__(state=state)
