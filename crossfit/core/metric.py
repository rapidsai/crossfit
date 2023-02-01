import abc
import inspect
from dataclasses import MISSING
from dataclasses import field as dataclass_field
from dataclasses import fields
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
from typing_extensions import dataclass_transform
from typing_utils import get_args, get_origin

from crossfit.array.conversion import ToType, convert
from crossfit.array.ops import concatenate
from crossfit.utils.types import get_generic_type_arg
from crossfit.dataframe.core import CrossFrame

OutputType = TypeVar("OutputType")
Array = TypeVar("Array", np.ndarray, pd.Series)


def field(
    *,
    combine=None,
    equals=None,
    is_list=False,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
):
    metadata = metadata or {}
    if combine is not None:
        if combine in (sum, "sum"):
            combine = _sum
        metadata["combine"] = combine
    if equals is not None:
        metadata["equals"] = equals
    if is_list:
        metadata["is_list"] = is_list

    return dataclass_field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        compare=compare,
        hash=hash,
        metadata=metadata,
    )


@runtime_checkable
class Combine(Protocol):
    def combine(self, other):
        ...


MetricT = TypeVar("MetricT", bound="Metric")


class State(Generic[MetricT]):
    ...


@dataclass_transform()
class MetricState(Generic[Array]):
    def combine(self, other):
        merged_fields = {}

        for field in fields(self):
            _type = get_origin(field.type)
            _type = _type or field.type

            if type(_type) != TypeVar and issubclass(_type, MetricState):
                # TODO: Add validation
                child = getattr(self, field.name)
                merged_fields[field.name] = child.combine(
                    getattr(other, field.name),
                )
            elif "combine" in field.metadata:
                merged_fields[field.name] = field.metadata["combine"](
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
            if field.type == Array:
                if isinstance(val, (int, float, str, list, tuple)):
                    object.__setattr__(self, field.name, self.to_state(val))

    def to_state(self, val) -> Array:
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

    def reduce(self):
        # Loop through fields, separating Array dependencies
        # from nested MetricState dependencies
        state_deps, array_deps, sizes = {}, {}, set()
        for field in fields(self):
            if field.type == Array:
                array_deps[field.name] = getattr(self, field.name)
                sizes.add(array_deps[field.name].shape[0])
            elif isinstance(getattr(self, field.name), MetricState):
                state_deps[field.name] = getattr(self, field.name).reduce()
            else:
                attr = getattr(self, field.name)
                raise ValueError(f"Expected Array or MetricState field, got {attr}")

        # Default logic requires all values in array_deps
        # to have the same length in the 0th dimension.
        # Otherwise, the MomentState subclass must define
        # a custom reduce method.
        if len(sizes) > 1:
            raise ValueError(
                f"Default reduce logic failed for {self.__class__}. "
                f"Field-array lengths do not all match: {sizes}."
            )
        size = list(sizes)[0] if len(sizes) else 1

        # Use combine logic to iteratively update a global state
        state = self.__class__(
            **{k: v[:1] for k, v in array_deps.items()},
            **state_deps,
        )
        for i in range(1, size):
            state = state.combine(
                self.__class__(
                    **{k: v[i : i + 1] for k, v in array_deps.items()},
                    **state_deps,
                )
            )
        return state

    def concat(self, *other, axis=None):
        params = {}
        for field in fields(self):
            if field.type == Array:
                params[field.name] = concatenate(
                    [
                        getattr(self, field.name),
                        *map(lambda x: getattr(x, field.name), other),
                    ],
                    axis=axis,
                )
            elif isinstance(getattr(self, field.name), MetricState):
                params[field.name] = getattr(self, field.name).concat(
                    *map(lambda x: getattr(x, field.name), other),
                    axis=axis,
                )
            else:
                attr = getattr(self, field.name)
                raise ValueError(f"Expected Array or MetricState field, got {attr}")

        return self.__class__(**params)

    @property
    def state_dict(self):
        output = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if field.type == Array:
                if "is_list" in field.metadata and val.ndim == 1:
                    output[field.name] = [val]
                else:
                    output[field.name] = val
            elif isinstance(getattr(self, field.name), MetricState):
                for key, child_val in getattr(self, field.name).state_dict.items():
                    output[".".join([field.name, key])] = child_val

        return output

    @classmethod
    def from_state(cls, state: dict):
        params = {}

        _state = unflatten_state(state)

        for field in fields(cls):
            _type = get_origin(field.type)
            _type = _type or field.type

            if _type == Array:
                params[field.name] = _state[field.name]
            elif isinstance(_type, type) and issubclass(_type, State):
                state_type = get_args(field.type)[0].state_type()
                params[field.name] = state_type.from_state(_state[field.name])
            elif isinstance(_type, type) and issubclass(_type, MetricState):
                params[field.name] = field.type.from_state(_state[field.name])

        return cls(**params)

    def state_df(self, adf: CrossFrame, index: Optional[str] = None):
        if not index:
            index = [
                self.cls_path(),
            ]
        return adf.from_dict(self.state_dict, index=list(index))

    @classmethod
    def from_state_df(cls, df: CrossFrame):
        return cls.from_state(df.to_dict())

    @classmethod
    def cls_path(cls):
        return ".".join([cls.__module__, cls.__name__])

    @classmethod
    def field_is_tensor(cls, field):
        return field.type == Array or (field.type == State and field.type.is_tensor())

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
    def prepare(self, data: Array, *args, **kwargs) -> StateType:
        raise NotImplementedError()

    def present(self, state: StateType) -> OutputType:
        return state

    def __call__(self, data: Array, present=True, **kwargs):
        state = self.prepare(data, **kwargs)

        if not present:
            return state

        return self.present(state)

    @classmethod
    def state_type(cls):
        prepare_type = inspect.signature(cls.prepare).return_annotation
        if prepare_type:
            return prepare_type
        cls_type = get_generic_type_arg(cls)
        if cls_type:
            return cls_type

        raise ValueError("Could not infer state type for {}".format(cls))


class AxisMetric(Metric[StateType], Generic[StateType], abc.ABC):
    def __init__(self, axis=0):
        self.axis = axis


class ComparisonMetric(Metric[StateType], Generic[StateType], abc.ABC):
    @abc.abstractmethod
    def prepare(self, data: Array, comparison: Array, **kwargs) -> StateType:
        raise NotImplementedError()

    def __call__(self, data: Array, comparison: Array, present=True, **kwargs):
        state = self.prepare(data, comparison, **kwargs)

        if not present:
            return state

        return self.present(state)


def _sum(left, right):
    return left + right
