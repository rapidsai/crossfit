from dataclasses import dataclass

from crossfit.core.metric import AxisMetric, MetricState, State
from crossfit.stats.common import CommonStats

# from crossfit.stats.continuous.histogram import Histogram
from crossfit.stats.continuous.moments import Moments
from crossfit.stats.continuous.range import Range


@dataclass
class ContinuousStatsState(MetricState):
    range: State[Range]
    moments: State[Moments]
    common: State[CommonStats]


class ContinuousStats(AxisMetric[ContinuousStatsState]):
    def prepare(self, data) -> ContinuousStatsState:
        return ContinuousStatsState(
            Range(axis=self.axis).prepare(data),
            Moments(axis=self.axis).prepare(data),
            CommonStats().prepare(data),
        )
