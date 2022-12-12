from dataclasses import dataclass

import crossfit.array as cnp
from crossfit.core.metric import Array, AxisMetric, MetricState, field


@dataclass
class RangeState(MetricState):
    min: Array = field(combine=cnp.minimum)
    max: Array = field(combine=cnp.maximum)


class Range(AxisMetric):
    def prepare(self, data) -> RangeState:
        return RangeState(data.min(axis=self.axis), data.max(axis=self.axis))


@dataclass
class AverageState(MetricState):
    count: Array = field(combine=sum)
    sum: Array = field(combine=sum)

    @property
    def average(self) -> Array:
        return self.sum / self.count


class Average(AxisMetric):
    def prepare(self, data) -> AverageState:
        return AverageState(len(data), data.sum(axis=self.axis))
