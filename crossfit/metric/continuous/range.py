import numpy as np

from crossfit.metric.base import CrossAxisMetric, state


class Range(CrossAxisMetric):
    min = state(init=0, combine=np.minimum)
    max = state(init=0, combine=np.maximum)

    def prepare(self, array) -> "Range":
        return Range(axis=self.axis, min=array.min(axis=self.axis), max=array.max(axis=self.axis))

    def present(self):
        return {"min": self.min, "max": self.max}
