# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft

import numpy as np

from crossfit.backend.dask.aggregate import aggregate
from crossfit.calculate.aggregate import Aggregator
from crossfit.metric.categorical.str_len import MeanStrLength
from crossfit.metric.categorical.value_counts import ValueCounts
from crossfit.metric.common import CommonStats
from crossfit.metric.continuous.moments import Moments
from crossfit.metric.continuous.range import Range
from crossfit.report.base import Report
from crossfit.report.data_overview.visualization.facets import FacetsOverview, visualize


class ContinuousMetrics(Aggregator):
    def prepare(self, array):
        return {
            "range": Range(axis=self.axis)(array),
            "moments": Moments(axis=self.axis)(array),
            "common_stats": CommonStats()(array),
        }


def is_continuous(col) -> bool:
    return np.issubdtype(col.dtype, np.number)


class CategoricalMetrics(Aggregator):
    def prepare(self, array):
        return {
            "value_counts": ValueCounts()(array),
            "mean_str_len": MeanStrLength()(array),
            "common_stats": CommonStats()(array),
        }


def is_categorical(col) -> bool:
    return col.dtype == object


class DataOverviewReport(Report):
    def __init__(self, con_df=None, cat_df=None):
        self.con_df = con_df
        self.cat_df = cat_df

    def visualize(self, name="data") -> FacetsOverview:
        return visualize(self.con_df, self.cat_df, name=name)


def column_select(df, col_fn):
    return [col for col in df.columns if col_fn(df[col])]


def data_overview_report(data, groupby=None) -> DataOverviewReport:
    continuous_agg = Aggregator(
        ContinuousMetrics(),
        per_column=ft.partial(column_select, col_fn=is_continuous),
        groupby=groupby,
    )
    categorical_agg = Aggregator(
        CategoricalMetrics(),
        per_column=ft.partial(column_select, col_fn=is_categorical),
        groupby=groupby,
    )

    con_df = aggregate(data, continuous_agg, to_frame=True)
    cat_df = aggregate(data, categorical_agg, to_frame=True)

    return DataOverviewReport(con_df, cat_df)
