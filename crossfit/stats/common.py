from dataclasses import dataclass

from crossfit.core.metric import Array, Metric, MetricState, field


@dataclass
class CommonStatsState(MetricState):
    count: Array = field(combine=sum)
    num_missing: Array = field(combine=sum)


class CommonStats(Metric):
    def prepare(self, data) -> CommonStatsState:
        return CommonStatsState(len(data), len(data[data.isnull()]))
