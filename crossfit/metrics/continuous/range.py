import numpy as np

from crossfit.metrics.base import CrossMetric, state


class Range(CrossMetric):
    min = state(init=0, combine=np.minimum)
    max = state(init=0, combine=np.maximum)

    def __init__(self, axis, **kwargs):
        self.axis = axis
        self.setup(**kwargs)

    def prepare(self, array) -> "Range":
        return Range(
            axis=self.axis, min=array.min(axis=self.axis), max=array.max(axis=self.axis)
        )

    def present(self):
        return {"min": self.min, "max": self.max}
