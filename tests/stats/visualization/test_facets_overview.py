import numpy as np

import crossfit as cf
from crossfit.core.calculate import calculate_per_col
from crossfit.stats.continuous.stats import ContinuousStats
from crossfit.stats.categorical.stats import CategoricalStats
from crossfit.stats.visualization import facets_overview
from tests.utils import sample_df


data = {
    "a": list(range(5)) * 2,
    "a2": list(range(5)) * 2,
    "b": np.random.rand(10),
    "c": np.random.rand(10),
    "cat": ["foo", "bar"] * 5,
}


@sample_df(data)
def test_facets_overview(df):
    con_cols = ["a", "a2", "b", "c"]
    cat_cols = ["cat"]

    con_mf: cf.MetricFrame = calculate_per_col(ContinuousStats(), df[con_cols])
    assert isinstance(con_mf, cf.MetricFrame)

    cat_mf = calculate_per_col(CategoricalStats(), df[cat_cols])
    assert isinstance(cat_mf, cf.MetricFrame)

    vis = facets_overview.visualize(con_mf=con_mf, cat_mf=cat_mf)
    assert isinstance(vis, facets_overview.FacetsOverview)


@sample_df(data)
def test_facets_overview_grouped(df):
    group_cols = ["a"]

    con_mf: cf.MetricFrame = calculate_per_col(
        ContinuousStats(), df[group_cols + ["b", "c"]], groupby=group_cols
    )
    cat_mf: cf.MetricFrame = calculate_per_col(
        CategoricalStats(), df[group_cols + ["cat"]], groupby=group_cols
    )

    vis = facets_overview.visualize(con_mf=con_mf, cat_mf=cat_mf)
    assert isinstance(vis, facets_overview.FacetsOverview)
