from dataclasses import dataclass

from crossfit.core.metric import AxisMetric, MetricState, State
from crossfit.stats.common import CommonStats

from crossfit.stats.categorical.common import ValueCounts, AverageStrLen


@dataclass
class CategoricalStatsState(MetricState):
    value_counts: State[ValueCounts]
    lengths: State[AverageStrLen]
    common: State[CommonStats]


class CategoricalStats(AxisMetric[CategoricalStatsState]):
    def prepare(self, data) -> CategoricalStatsState:
        return CategoricalStatsState(
            ValueCounts(axis=self.axis).prepare(data),
            AverageStrLen(axis=self.axis).prepare(data),
            CommonStats().prepare(data),
        )

    def present(self, state: CategoricalStatsState):
        output = state.value_counts.top_k()
        output["str_len"] = state.lengths.average
        for key, val in state.common.state_dict.items():
            output["common." + key] = val
        output["num_unique"] = len(state.value_counts.values)

        return output
