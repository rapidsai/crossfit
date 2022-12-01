from dataclasses import dataclass

import numpy as np

from crossfit.core.metric import Metric, StateType, OutputType
from crossfit.core.state import MetricState, ArrayType


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
