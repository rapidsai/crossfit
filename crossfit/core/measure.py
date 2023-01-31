from __future__ import annotations

from dataclasses import dataclass, replace

from typing import Any, List
from collections import defaultdict


from crossfit.core.frame import MetricFrame
from crossfit.core.metric import Metric, ComparisonMetric
from crossfit.dataframe.core import AbstractFrame
from crossfit.dataframe.dispatch import frame_dispatch


def measure(
    metric: Metric,
    data: Any,
    groupby: List[str] | None = None,
    compare_columns: List[str] | None = None,
):
    # TODO: Implement this logic as `Metric.apply`?
    # TODO: Can use a tree-reduction when data is a Dask collection
    # Convert input data to an abstract-dataframe representation
    # (Perhaps create a lazy DaskMeasurement object?)
    adf = frame_dispatch(data)

    if isinstance(metric, ComparisonMetric):
        assert compare_columns
        if groupby:
            return GroupedComparison.prepare(adf, metric, groupby, compare_columns)
        return Comparison.prepare(adf, metric, compare_columns)

    assert not compare_columns
    if groupby:
        return GroupedMeasurement.prepare(adf, metric, groupby)
    return Measurement.prepare(adf, metric)


@dataclass(frozen=True)
class Measurement:
    """Metric-Measurement class
    Tracks the state of a dataset-metric measurement
    """

    metric: Metric
    state: dict
    adf_type: Any

    @classmethod
    def prepare(cls, data: AbstractFrame, metric: Metric):
        state = {col: metric.prepare(data.select_column(col)) for col in data.columns}
        return cls(metric, state, type(data))

    @classmethod
    def combine(cls, measurements: list):
        col_states = defaultdict(list)
        for measurement in measurements:
            for col, col_state in measurement.state.items():
                col_states[col].append(col_state)
        state = {}
        for col, state_list in col_states.items():
            state[col] = state_list[0].concat(*state_list[1:]).reduce()
        return replace(measurements[0], state=state)

    def finalize(self) -> MetricFrame:
        # Loop over columns
        rows = []
        index = []
        for name_col, col_state in self.state.items():
            state_df = col_state.state_df(self.adf_type)
            index.append(name_col)
            rows.append(state_df)

        # Return a MetricFrame summary
        return MetricFrame(
            self.adf_type.concat(rows, axis=0, ignore_index=True),
            metric=self.metric,
            index=self.adf_type.from_dict({"col": index}).select_column("col"),
        )


@dataclass(frozen=True)
class Comparison(Measurement):
    columns: list

    @classmethod
    def prepare(cls, data: AbstractFrame, metric: Metric, columns: list):
        assert isinstance(columns, list)
        state = {
            tuple(columns): metric.prepare(
                *[data.select_column(col) for col in columns]
            )
        }
        return cls(metric, state, type(data), columns)


@dataclass(frozen=True)
class GroupedMeasurement(Measurement):
    """Grouped metric-Measurement class
    Tracks the state of a dataset-metric Measurement
    """

    groupby: List[str]

    @classmethod
    def prepare(cls, data: AbstractFrame, metric: Metric, groupby: list):
        state = {}
        groups = data.groupby_partition(groupby)
        for slice_key, group_df in groups.items():
            if not isinstance(slice_key, tuple):
                slice_key = (slice_key,)
            state[slice_key] = {}
            for name_col in group_df.columns:
                if name_col in groupby:
                    continue
                state[slice_key][name_col] = metric.prepare(
                    group_df.select_column(name_col)
                )
        return cls(metric, state, type(data), groupby)

    @classmethod
    def combine(cls, measurements: list):
        all_states = {}
        for measurement in measurements:
            for slice_key, slice_states in measurement.state.items():
                if slice_key not in all_states:
                    all_states[slice_key] = defaultdict(list)
                for col_name, state in slice_states.items():
                    all_states[slice_key][col_name].append(state)
        states = {}
        for slice_key, column_states in all_states.items():
            states[slice_key] = {}
            for col_name, state_list in column_states.items():
                states[slice_key][col_name] = (
                    state_list[0].concat(*state_list[1:]).reduce()
                )
        return replace(measurements[0], state=states)

    def finalize(self) -> MetricFrame:
        # Loop over columns
        rows = []
        index = []
        cols = defaultdict(list)
        for slice_key, column_states in self.state.items():
            for name_col, state in column_states.items():
                if name_col in self.groupby:
                    continue
                state_df = state.state_df(self.adf_type)
                index.append(name_col)
                rows.append(state_df)
                cols["col"].append(name_col)
                for i, k in enumerate(self.groupby):
                    cols[k].append(slice_key[i])

        # Return a MetricFrame summary
        return MetricFrame(
            self.adf_type.concat(rows, axis=0, ignore_index=True),
            metric=self.metric,
            data=self.adf_type.from_dict(cols),
        )


@dataclass(frozen=True)
class GroupedComparison(Measurement):
    columns: list

    @classmethod
    def prepare(cls, data: AbstractFrame, metric: Metric, groupby: list, columns: list):
        state = {}
        groups = data.groupby_partition(groupby)
        for slice_key, group_df in groups.items():
            if not isinstance(slice_key, tuple):
                slice_key = (slice_key,)
            state[slice_key] = {
                tuple(columns): metric.prepare(
                    *[group_df.select_column(col) for col in columns]
                )
            }
        return cls(metric, state, type(data), groupby, columns)
