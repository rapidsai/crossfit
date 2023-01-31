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
        combined_frame = (
            self_series.to_frame()
            .join(
                other_series.to_frame(),
                lsuffix="_l",
                rsuffix="_r",
                how="outer",
            )
            .fillna(0)
        )
        combined = (combined_frame["count_l"] + combined_frame["count_r"]).astype(
            "int64"
        )
        return ValueCountsState(combined.index, combined.values)

    def top_k(self, k=10) -> Dict[str, int]:
        counts = {"top_values": [], "top_counts": []}
        # TODO: This doesn't actually return
        # "top k", just the first k unique items
        # (in an undetermined order)
        if hasattr(self.values, "to_pandas"):
            _values = self.values.to_pandas()
            _counts = self.counts.to_pandas()
        else:
            _values = self.values
            _counts = self.counts

        if _values.dtype == object:
            for i, val in enumerate(_values):
                counts["top_values"].append(val[:k])
                counts["top_counts"].append(_counts[i][:k])
        else:
            counts["top_counts"] = _counts[:k]
            counts["top_values"] = _values[:k]
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
