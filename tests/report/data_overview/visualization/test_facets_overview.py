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

# import numpy as np

# import crossfit as cf
# from crossfit.calculate.calculate import calculate_per_col
# from crossfit.stats.continuous.stats import ContinuousStats
# from crossfit.stats.categorical.stats import CategoricalStats
# from crossfit.stats.visualization import facets_overview
# from tests.utils import sample_df


# data = {
#     "a": list(range(5)) * 2,
#     "a2": list(range(5)) * 2,
#     "b": np.random.rand(10),
#     "c": np.random.rand(10),
#     "cat": ["foo", "bar"] * 5,
# }


# @sample_df(data)
# def test_facets_overview(df):
#     con_cols = ["a", "a2", "b", "c"]
#     cat_cols = ["cat"]

#     con_mf: cf.MetricFrame = calculate_per_col(ContinuousStats(), df[con_cols])
#     assert isinstance(con_mf, cf.MetricFrame)

#     cat_mf = calculate_per_col(CategoricalStats(), df[cat_cols])
#     assert isinstance(cat_mf, cf.MetricFrame)

#     vis = facets_overview.visualize(con_mf=con_mf, cat_mf=cat_mf)
#     assert isinstance(vis, facets_overview.FacetsOverview)


# @sample_df(data)
# def test_facets_overview_grouped(df):
#     group_cols = ["a"]

#     con_mf: cf.MetricFrame = calculate_per_col(
#         ContinuousStats(), df[group_cols + ["b", "c"]], groupby=group_cols
#     )
#     cat_mf: cf.MetricFrame = calculate_per_col(
#         CategoricalStats(), df[group_cols + ["cat"]], groupby=group_cols
#     )

#     vis = facets_overview.visualize(con_mf=con_mf, cat_mf=cat_mf)
#     assert isinstance(vis, facets_overview.FacetsOverview)
