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

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import crossfit as cf
from crossfit.backend.dask.aggregate import aggregate
from crossfit.report.data_overview.report import (
    CategoricalMetrics,
    ContinuousMetrics,
    DataOverviewReport,
    data_overview_report,
)
from crossfit.report.data_overview.visualization.facets import FacetsOverview
from tests.utils import sample_df


@sample_df({"a": [1, 2] * 2000, "b": range(1000, 5000)})
def test_continuous_aggregators(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)

    metrics = cf.Aggregator(ContinuousMetrics(), per_column=True)

    result = aggregate(ddf, metrics, to_frame=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 7


@pytest.mark.skip(reason="Not implemented for pyarrow[string] yet")
@sample_df(
    {
        "a": np.random.choice(list("abcdefgh"), size=1000),
        "country": np.random.choice(["US", "UK", "NL"], size=1000),
    }
)
def test_categorical_aggregator(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)

    metrics = cf.Aggregator(CategoricalMetrics(), per_column=True)

    result = aggregate(ddf, metrics, to_frame=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 6


@pytest.mark.skip(reason="Not implemented for pyarrow[string] yet")
@sample_df(
    {
        "con": [1, 2] * 500,
        "country": np.random.choice(["US", "UK", "NL"], size=1000),
    }
)
def test_data_overview_report(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)

    report = data_overview_report(ddf)

    assert isinstance(report, DataOverviewReport)

    visualization = report.visualize()
    assert isinstance(visualization, FacetsOverview)


@pytest.mark.skip(reason="Not implemented for pyarrow[string] yet")
@sample_df(
    {
        "con": [1, 2] * 500,
        "cat": ["a", "b"] * 500,
        "country": np.random.choice(["US", "UK", "NL"], size=1000),
    }
)
def test_data_overview_report_grouped(df, npartitions=2):
    ddf = dd.from_pandas(df, npartitions=npartitions)

    report = data_overview_report(ddf, groupby="country")

    assert isinstance(report, DataOverviewReport)

    visualization = report.visualize()
    assert isinstance(visualization, FacetsOverview)
