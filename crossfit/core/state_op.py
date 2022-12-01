import abc
import sys
from typing import TypeVar, Generic, Optional, Dict, Protocol, runtime_checkable, Union
from dataclasses import dataclass

from dask.utils import Dispatch
from classes import typeclass
import pandas as pd
import numpy as np
import cudf as cd
from scipy.stats import moment

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")


class CalculateDispatch(Dispatch):
    def __call__(self, input, **kwargs) -> StateT:
        return super().__call__(input, **kwargs)


def get_generic_type_arg(cls):
    t = cls.__orig_bases__[0]
    py_version = sys.version_info
    if py_version >= (3, 8):
        from typing import get_args
        return get_args(t)[0]
    else:
        return t.__args__[0]


class Metric(Generic[StateT]):
    def __init__(
        self, 
        data=None,
        index=None,
        pre=None,
        post=None,
        state: Optional[StateT] = None,
        **kwargs
    ):
        self.index = index
        self.pre = pre
        self.post = post
        self.kwargs = kwargs
        
        if data is not None:
            state = self.calculate(data)
            
        if state is not None:
            self.validate_state(state)
            self.state = state
    
    def calculate(self, data) -> StateT:
        if self.pre:
            data = self.pre(data)
        return self.calculate_state(data)                   
            
    @abc.abstractmethod
    def calculate_state(self, data) -> StateT:
        raise NotImplementedError

    def validate_state(self, state):
        assert state is not None
        
    def result(self) -> OutputT:
        return self.state
    
    def concat(self, other):
        if hasattr(self.state, "concat"):
            state = self.state.concat(other.state)
        else:
            state = np.array([self.state, other.state])
            
        index = None
        if self.index and other.index:
            # self_index = list(self.index) if not isinstance(self.index, list) else self.index
            # other_index = list(other.index) if not isinstance(other.index, list) else other.index
            # index = np.concatenate((self_index + other_index))
            index = list(self.index) + list(other.index)
        
        return self.__class__(
            state=state,
            pre=self.pre,
            post=self.post,
            index=index,
            **self.kwargs
        )
    
    def __add__(self, other):
        return self.merge(other)
    
    def __truediv__(self, other):
        if isinstance(other, Metric):
            other = other.state
            
        return self.state / other
    
    def __len__(self):
        return len(self.state)
    
    def __call__(self, data, pre=None, post=None, index=None):
        state = self.calculate(data)
        
        return self.__class__(
            state=state,
            pre=pre or self.pre,
            post=post or self.post,
            index=index or self.index,
            **self.kwargs
        )
    
    # def __repr__(self):
    #     state = self.state
    #     if len(state) == 1:
    #         state_str = state[0]
    #     else:
    #         state_str = ", ".join(f"{k}={v}" for k, v in self.state_dict().items())
    #     return f"{self.__class__.__name__}({state_str})"
    

def register(cls, type, func=None):
    def _register(func):
        cls.dispatch.register(type, func=func)
        return func
    return _register
    
    
class MetricTypeClass(Metric):
    dispatch = CalculateDispatch("calculate")
    
    def calculate_state(self, data) -> StateT:
        return self.dispatch(data, **self.kwargs)
    
    
class Sum(MetricTypeClass):
    def merge(self, other) -> "Sum":
        return Sum(self.state + other.state)
    

@register(Sum, np.ndarray)
@register(Sum, pd.Series)
def sum_np(array, **kwargs) -> np.ndarray:
	return array.sum(**kwargs)
    
  
@typeclass
def count(data):
    ...  

    
class Count(Metric):
    def calculate_state(self, data) -> StateT:
        return count(data)
    
    def merge(self, other) -> "Sum":
        return Count(self.state + other.state)


@runtime_checkable
class LenProtocol(Protocol):
    def __len__(self):
        ...


@count.instance(protocol=LenProtocol)
def count_len_protocol(data) -> int:
    return len(data)
    

class MetricState:
    def items(self):
        raise NotImplementedError()
    
    
    

@dataclass    
class MeanState(MetricState):
    sum: Sum
    count: Count
    
    def concat(self, other):
        return MeanState(
            sum=self.sum.concat(other.sum),
            count=self.count.concat(other.count)
        )
  
    
class Mean(Metric[MeanState]):
    def calculate_state(self, data) -> MeanState:
        return MeanState(Sum(data, **self.kwargs), Count(data))
    
    def compute(self):
        return self.state.sum / self.state.count
    
    def state_df(self):
        df = pd.DataFrame({"sum": self.state.sum.state, "count": self.state.count.state})
        df.attrs["cls"] = self.__class__
        
        return df
    
    @classmethod
    def from_state(cls, state):
        return cls(state=MeanState(Sum(state=state["sum"]), Count(state=state["count"])))


ArrayLike = TypeVar("ArrayLike", np.ndarray, list, tuple)


class Moments(Metric[ArrayLike]):
    def calculate_state(self, data) -> ArrayLike:
        return moment(data, np.array([0,1,2,3]), **self.kwargs)



StateT = TypeVar("StateT", bound=Metric)


class StateFrame(Generic[StateT]):
    def __init__(
        self, 
        data=None,
        state: Optional[Union[StateT, Dict[str, StateT]]] = None,
        index=None,
    ):
        self.data = data
        self.state = state.state_df()
        self.index = index
        
    def result(self):
        out = pd.DataFrame(self.data)
        state = self.state.attrs["cls"].from_state(self.state)
        out["result"] = state.compute()
        
        return out
    
    
    
def calculate_df(df, metric: Metric):
    out = None
    for name, col in df.items():
        m = metric(col, index=name) 
        if out is None:
            out = m
        else:
            out = out.concat(m)
    
    return out


def calculate_cols(df, metric: Metric):
    out = []
    for name, col in df.items():
        out.append(metric(col, index=name) )
    
    return out


def calculate_grouped(df, groups, metric: Metric):
    grouped = df.groupby(groups)
    
    slices = {}
    for name, slice in dict(grouped.groups).items():
        slices[name] = calculate_df(df.iloc[slice], metric)
        
        
    return slices


if __name__ == '__main__':
    arr_1d = np.array(list(range(10)))
    arr_2d = np.random.rand(10,10)
    df = pd.DataFrame({"a": list(range(5)) * 2, "b": list(range(10))})
    df2 = cd.DataFrame({"a": list(range(5)) * 2, "b": list(range(10))})
    
    # test = Sum(arr_2d, axis=0)
    # c = Count(arr_1d)
    # mom = Moments(arr_1d)
    mean = Mean(arr_2d, axis=0).state_df()
    
    m = calculate_df(df, Mean())
    cols = calculate_cols(df, Mean())
    m_grouped = calculate_grouped(df, "a", Mean())
    
    sdf = StateFrame(data=df, index="a", state=Mean(arr_2d, axis=0))
    
    a = 5