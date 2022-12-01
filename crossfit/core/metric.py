import abc
from typing import Generic, TypeVar, Optional

from crossfit.core.op import Op


class MergeMixin:
    @abc.abstractmethod
    def merge(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        return self.merge(other)


ResultT = TypeVar("ResultT")


class ResultMixin(abc.ABC, Generic[ResultT]):
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def result(self) -> ResultT:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

        return self


class Metric(Op, MergeMixin):
    def update(self, *args, **kwargs):
        self._update_children(*args, **kwargs)
    
    def result(self) -> ResultT:
        return self._result_children()

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

        return self

    def _update_children(self, *args, **kwargs):
        for child in self.children():
            child.update(*args, **kwargs)
   
    def _result_children(self):
        output = {}
        for field in self.fields():
            value = getattr(self, field.name)
            if isinstance(value, Op):                
                child_result = value.result()
                if isinstance(child_result, dict):
                    for name, val in child_result.items():
                        output[".".join([field.name, name])] = val
                else:
                    output[field.name] = child_result
        
        return output


class ArrayMetric(Metric, abc.ABC):
    @abc.abstractmethod	
    def update(self, array):
        raise NotImplementedError()
    
    


class WeightedArrayMetric(Metric, abc.ABC):
    @abc.abstractmethod	
    def update(self, array, weigths=None):
        raise NotImplementedError()


# class PredictionMetric(Metric):
# 	def update(
# 		self, 
# 		prediction, 
# 		target, 
# 		weights=None
# 	):
# 		...


# class ComparisonMetric(Metric):
# 	def update(self, left, right):
# 		...
  