import numpy as np

from crossfit.metrics.base import CrossMetric, state


class Range(CrossMetric):
    min = state(init=0, combine=np.min)
    max = state(init=0, combine=np.max)

    def prepare(self, array, *, axis) -> "Range":
        return Range(min=array.min(axis=axis), max=array.max(axis=axis))
