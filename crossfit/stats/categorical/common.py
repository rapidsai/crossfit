from typing import Dict
from dataclasses import dataclass

import crossfit.array as cnp

from crossfit.core.metric import Array, AxisMetric, MetricState, field
from crossfit.stats.continuous.common import AverageState
from crossfit.dataframe.dispatch import frame_dispatch


class AverageStrLen(AxisMetric):
    def prepare(self, data: Array) -> AverageState:
        return AverageState(len(data), data.str.len().sum(axis=self.axis))


@dataclass
class ValueCountsState(MetricState):
    values: Array = field(is_list=True)
    counts: Array = field(is_list=True)

    def combine(self, other: "ValueCountsState") -> "ValueCountsState":
        self_series = frame_dispatch(
            self.counts,
            column_name="count",
            index=self.values,
        ).select_column("count")
        other_series = frame_dispatch(
            other.counts,
            column_name="count",
            index=other.values,
        ).select_column("count")
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
            d = data.value_counts().sort_index()
            values = d.index
            counts = d.values
        else:
            values, counts = cnp.unique(data, axis=self.axis, return_counts=True)
        return ValueCountsState(values, counts)

    def present(self, state: ValueCountsState) -> Dict[str, int]:
        return state.top_k()
