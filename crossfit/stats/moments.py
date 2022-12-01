import abc
import inspect
import sys
from dataclasses import dataclass, fields
from typing import Generic, TypeVar, Optional, Type

import numpy as np
import pandas as pd

from crossfit.dispatch.array import convert, ToType




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
            elif(isinstance(val, MetricState)):
                params[field.name] = val.convert(type)
            else:
                params[field.name] = convert(val, type)
        
        return self.__class__(**params)
    
    def concat(self, *other, axis=None):
        params = {}
        for field in fields(self):
            if field.type == ArrayType:
                params[field.name] = np.concatenate([
                    getattr(self, field.name), 
                    *map(lambda x: getattr(x, field.name), other)
                ], axis=axis)
                
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
            index = [self.cls_path(),]
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


# Adapted from: https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/wrappers/normalize.py#L25
@dataclass(frozen=True)
class MomentsState(MetricState[ArrayType]):
    count: ArrayType
    mean: ArrayType
    var: ArrayType

    def merge(self, other: "MomentsState") -> "MomentsState":
        delta = other.mean - self.mean
        tot_count = self.count + other.count

        new_mean = self.mean + delta * other.count / tot_count
        m_self = self.var * self.count
        m_other = other.var * other.count
        M2 = m_self + m_other + (delta ** 2) * self.count * other.count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return MomentsState(count=new_count, mean=new_mean, var=new_var)
    
    @property
    def variance(self):
        return self.var

    @property
    def std(self):
        return np.sqrt(self.variance)


def moments(data: ArrayType, axis=0) -> MomentsState[ArrayType]:
    return MomentsState(
        count=data.shape[axis],
        mean=data.mean(axis=axis),
        var=data.var(axis=axis),
    )
    
    

@dataclass(frozen=True)
class Min(MetricState):
    state: ArrayType
    
    def merge(self, other):
        return Min(np.minimum(self.state, other.state))
    
    
@dataclass(frozen=True)
class Max(MetricState):
    state: ArrayType
    
    def merge(self, other):
        return Max(np.maximum(self.state, other.state))
    
    
@dataclass(frozen=True)
class Sum(MetricState):
    state: ArrayType
    
    def merge(self, other):
        return Sum(self.state + other.state)
    
    
@dataclass(frozen=True)    
class MeanState(MetricState):
    count: ArrayType
    total: ArrayType
    
    def merge(self, other):
        return Mean(
            count=self.count + other.count,
            total=self.total + other.total,
        )
        
    @property
    def mean(self):
        return self.total / self.count

    
    
StateType = TypeVar("StateType", bound=MetricState, covariant=True)
OutputType = TypeVar("OutputType")


def get_generic_type_arg(cls):
    t = cls.__orig_bases__[0]
    if sys.version_info >= (3, 8):
        from typing import get_args
        return get_args(t)[0]
   
    return t.__args__[0]


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
    
    
    
class Moments(Metric[MomentsState]):
    def __init__(self, axis=0):
        self.axis = axis
    
    def prepare(self, data: ArrayType) -> MomentsState:
        return MomentsState(
            count=data.shape[self.axis],
            mean=data.mean(axis=self.axis), 
            var=data.var(axis=self.axis),
        )
        
    def present(self, state: StateType) -> OutputType:
        return {"mean": state.mean, "variance": state.variance}
    
    
class Mean(Metric):
    def __init__(self, axis=0):
        self.axis = axis
        
    def prepare(self, data: ArrayType) -> MeanState:
        return MeanState(data.sum(axis=self.axis), data.shape[self.axis])
    
    def present(self, state: StateType) -> OutputType:
        return state.mean
    
    
import tensorflow as tf


class TFMetric(ComparisonMetric):
    def to_mean(self, metric: tf.keras.metrics.Metric, array_type: ArrayType = np.ndarray) -> Mean:
        return Mean(
            count=convert(tf.convert_to_tensor(metric.count), array_type),
            total=convert(tf.convert_to_tensor(metric.total), array_type),
        )


class TFAccuracy(TFMetric):
    def prepare(self, data: ArrayType, comparison: ArrayType, **kwargs) -> Mean:
        acc = tf.keras.metrics.Accuracy()
        acc.update_state(
            convert(data, tf.Tensor),
            convert(comparison, tf.Tensor)
        )
        
        return self.to_mean(acc, array_type=type(data))
    
    def present(self, state: Mean) -> ArrayType:
        return state.mean
    
    
    
class TFAUC(TFMetric):
    def prepare(self, data: ArrayType, comparison: ArrayType, **kwargs) -> Mean:
        auc = tf.keras.metrics.AUC()
        auc.update_state(
            convert(data, tf.Tensor),
            convert(comparison, tf.Tensor)
        )
        
        return self.to_mean(auc, array_type=type(data))
    
    def present(self, state: Mean) -> ArrayType:
        return state.mean
    
    
    
class MetricFrame:
    def __init__(self, state_df, metric=None, data=None, index=None):
        if not metric:
            if not "cls" in state_df.attrs:
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
        if not isinstance(metric_result, dict):
            metric_result = {"result": metric_result}
            
        result_df = pd.DataFrame(metric_result, index=self.index)
            
        if self.data is not None:
            df = pd.concat([self.data, pd.DataFrame(metric_result)], axis=1)
            df = df.pivot(index=set(self.data.columns) - set(["col"]), columns=["col"])
            
            return df
        
        return result_df
        


def calculate(data, metric: Metric):
    # This could be backed by a dispatch-type approach with some different types of data
    pass


def calculate_cols(df, metric: Metric):
    out = []
    for name, col in df.items():
        out.append(metric(col, index=name) )
    
    return out


def calculate_grouped(df_grouped, metric: Metric):
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        index.append(slice_key)
        cols = []
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col)
            state_df = state.state_df()
            state_df.columns = pd.MultiIndex.from_product([[name_col], list(state.state_dict.keys())])
            cols.append(state_df)
            
        rows.append(pd.concat(cols, axis=1))
    if len(df_grouped.keys) > 1:
        pd_index = pd.MultiIndex.from_tuples(index, names=df_grouped.keys)
    else:
        pd_index = pd.Index(index, name=df_grouped.keys)
        
    output = pd.concat(rows, axis=0)
    output.index = pd_index
    
    return output


def calculate_state_grouped(df_grouped, metric: Metric):
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col)
            state_df = state.state_df()
            index.append((name_col,) + slice_key)
            
            rows.append(state_df)
    if len(df_grouped.keys) > 1:
        pd_index = pd.MultiIndex.from_tuples(index, names=["col"] + df_grouped.keys)
    else:
        pd_index = pd.Index(index, name=df_grouped.keys)
        
    output = pd.concat(rows, axis=0)
    output.index = pd_index
    
    return output


def _calculate_grouped_per_col(df_grouped, metric: Metric) -> MetricFrame:
    index = []
    rows = []
    for slice_key, slice in dict(df_grouped.groups).items():
        for name_col, col in df_grouped.obj.iloc[slice].items():
            if name_col in df_grouped.keys:
                continue
            state = metric.prepare(col)
            state_df = state.state_df()
            index.append((name_col,) + slice_key)
            
            rows.append(state_df)
    if len(df_grouped.keys) > 1:
        pd_index = pd.MultiIndex.from_tuples(index, names=["col"] + df_grouped.keys)
    else:
        pd_index = pd.Index(index, name=df_grouped.keys)
        
    df = pd.concat(rows, axis=0)
    df.index = pd_index
    
    df = df.reset_index()
    cols = df[pd_index.names]
    state_df = df[set(df.columns) - set(pd_index.names)]    
    mdf = MetricFrame(state_df, metric=metric, data=cols)
    
    return mdf


def calculate_per_col(df, metric: Metric) -> MetricFrame:
    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        return _calculate_grouped_per_col(df, metric)
    
    rows = []
    index = []
    for name_col, col in df.items():
        state = metric.prepare(col)
        state_df = state.state_df()
        index.append(name_col)
        rows.append(state_df)
        
    df = pd.concat(rows, axis=0)    
    mdf = MetricFrame(df, metric=metric, index=pd.Index(index, name="col"))
    
    return mdf
    

if __name__ == "__main__":
    array = np.array([1, 2, 3, 4, 5])
    df = pd.DataFrame({"a": list(range(5)) * 2, "a2": list(range(5)) * 2, "b": np.random.rand(10), "c": np.random.rand(10)})
    series = pd.Series(array)
    
    metric = Moments()
    mdf = calculate_per_col(df, metric)
    mdf_grouped = calculate_per_col(df.groupby(["a", "a2"]), metric)
    
    a = 5
    
    tf_array = np.array([1] * 10)
    acc = TFAccuracy().prepare(tf_array, tf_array)
    auc = TFAUC().prepare(tf_array, tf_array)
    
    
    
    
    a = 5