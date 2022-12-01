from crossfit.core.metric import ArrayMetric

import numpy as np


class Max:
    def __init__(self, data=float("-inf")):
        self.val = data
        
    def merge(self, other):
        return Max(max(self.val, other.val))


class Min(ArrayMetric):
    val: float = float("inf")
    
    def update(self, array):
        self.update_single(array.min())
        
    def update_single(self, val):
        self.val = min(self.val, val)
    
    def result(self):
        return self.val
    
    def merge(self, other):
        return Min(min(self.val, other.val))
    
    
class Max(ArrayMetric):
    val: float = float("-inf")
    
    def update(self, array):
        self.update_single(array.max())
        
    def update_single(self, val):
        self.val = max(self.val, val)
    
    def result(self):
        return self.val
    
    def merge(self, other):
        self.update_single(other.val)
        return self
    
    
class Range(ArrayMetric):
    min: Min = Min()
    max: Max = Max()