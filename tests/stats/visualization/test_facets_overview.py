import numpy as np
import pandas as pd

import crossfit as cf
from crossfit.core.calculate import calculate_per_col
from crossfit.stats.continuous.stats import ContinuousStats
from crossfit.stats.categorical.stats import CategoricalStats
from crossfit.stats.visualization import facets_overview


df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
        "cat": ["foo", "bar"] * 5,
    }
)


def test_facets_overview():
    con_cols = ["a", "a2", "b", "c"]
    cat_cols = ["cat"]

    con_mf: cf.MetricFrame = calculate_per_col(df[con_cols], ContinuousStats())
    assert isinstance(con_mf, cf.MetricFrame)

    cat_mf = calculate_per_col(df[cat_cols], CategoricalStats())
    assert isinstance(cat_mf, cf.MetricFrame)

    vis = facets_overview.visualize(con_mf=con_mf, cat_mf=cat_mf)
    assert isinstance(vis, facets_overview.FacetsOverview)
