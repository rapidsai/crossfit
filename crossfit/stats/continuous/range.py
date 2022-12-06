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
