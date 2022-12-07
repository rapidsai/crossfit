import numpy as np
import pandas as pd

import crossfit as cf
from crossfit.core.calculate import calculate_per_col
from crossfit.stats.continuous.stats import ContinuousStats
from crossfit.stats.visualization import facets_overview


df = pd.DataFrame(
    {
        "a": list(range(5)) * 2,
        "a2": list(range(5)) * 2,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
    }
)


def test_facets_overview():
    mf: cf.MetricFrame = calculate_per_col(df, ContinuousStats())
    assert isinstance(mf, cf.MetricFrame)

    vis = facets_overview.visualize(mf)
    assert isinstance(vis, facets_overview.FacetsOverview)
