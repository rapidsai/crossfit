from typing import Dict
from dataclasses import dataclass

import crossfit.array as cnp
import pandas as pd

from crossfit.core.metric import Array, AxisMetric, MetricState, field
from crossfit.stats.continuous.common import AverageState


class AverageStrLen(AxisMetric):
    def prepare(self, data: Array) -> AverageState:
        return AverageState(len(data), data.str.len().sum(axis=self.axis))


@dataclass
class ValueCountsState(MetricState):
    values: Array = field(is_list=True)
    counts: Array = field(is_list=True)

    def combine(self, other: "ValueCountsState") -> "ValueCountsState":
        # TODO: Make this framework agnostic
        self_series = pd.Series(self.counts, index=self.values)
        other_series = pd.Series(other.counts, index=other.values)

        combined = self_series + other_series

        return ValueCountsState(combined.index.values, combined.values)

    def top_k(self, k=10) -> Dict[str, int]:
        counts = {"top_values": [], "top_counts": []}

        if self.values.dtype == object:
            for i, val in enumerate(self.values):
                counts["top_values"].append(val[:k])
                counts["top_counts"].append(self.counts[i][:k])
        else:
            counts["top_counts"] = self.counts[:k]
            counts["top_values"] = self.values[:k]

        return counts


class ValueCounts(AxisMetric):
    def prepare(self, data: Array) -> ValueCountsState:
        if hasattr(data, "value_counts"):
            d = data.value_counts()
            values = d.index.values
            counts = d.values
        else:
            values, counts = cnp.unique(data, axis=self.axis, return_counts=True)
        return ValueCountsState(values, counts)

    def present(self, state: ValueCountsState) -> Dict[str, int]:
        return state.top_k()
