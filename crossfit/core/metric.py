import abc
import inspect
import sys
from typing import Generic, TypeVar

from crossfit.core.state import ArrayType, MetricState

StateType = TypeVar("StateType", bound=MetricState, covariant=True)
OutputType = TypeVar("OutputType")


class Metric(Generic[StateType], abc.ABC):
    @abc.abstractmethod
    def prepare(self, data: ArrayType, *args, **kwargs) -> StateType:
        raise NotImplementedError()

    @abc.abstractmethod
    def present(self, state: StateType) -> OutputType:
        raise NotImplementedError()

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


def get_generic_type_arg(cls):
    t = cls.__orig_bases__[0]
    if sys.version_info >= (3, 8):
        from typing import get_args

        return get_args(t)[0]

    return t.__args__[0]
