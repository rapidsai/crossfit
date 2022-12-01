from crossfit.core.op import Op


class ResultMixin:
    def result(self):
        return self.data
    
    
    
def with_dispatch(fn):
    return fn


@with_dispatch
class Max(Op, ResultMixin):
    data: float = float("-inf")
    
    def merge(self, other) -> "Max":
        return Max(max(self.val, other.val))
    
    
class Min(Op, ResultMixin):
    data: float = float("inf")
    
    def merge(self, other) -> "Min":
        return Min(min(self.val, other.val))
    
    
class Sum(Op, ResultMixin):
    data: float = 0.0
    
    def merge(self, other) -> "Sum":
        return Sum(self.val + other.val)
    
    
class Count(Sum):
    def __init__(self, array):
        super().__init__(len(array))
    
    
class Range(OpCollection):
    min: Min
    max: Max
    

  
    
r = Range([1,2,3,4])
    
    
def min(data) -> Min:
    ...
    
    
def max(data) -> Max:
    ...
    
    
