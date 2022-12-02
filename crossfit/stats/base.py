from dataclasses import dataclass

from crossfit.core.metric import Metric, MetricState, State
from crossfit.dispatch.combiner import Max, Min


@dataclass
class RangeState(MetricState):
    min: State[Min]
    max: State[Max]


class Range(Metric):
    def __init__(self, axis=0):
        self.axis = axis

    def prepare(self, data) -> RangeState:
        return RangeState(data.min(axis=self.axis), data.max(axis=self.axis))
