import pandas as pd
import numpy as np

from crossfit.core.state import State, PerColumn, Stats, Grouped
from crossfit.stats.continuous.stats import ContinuousStats, ContinuousStatsState


df = pd.DataFrame(
    {
        "name": ["train", "test"] * 5,
        "con_1": np.random.rand(10),
        "con_2": np.random.rand(10),
        "cat_1": ["foo", "bar"] * 5,
    }
)


def test_per_col():
    con_df = df[["con_1", "con_2"]]
    con_per_col = PerColumn(ContinuousStats())(con_df)

    assert isinstance(con_per_col, State)
    assert isinstance(con_per_col.obj(), ContinuousStatsState)

    assert isinstance(con_per_col.moments.mean, np.ndarray)


def test_stats():
    stats = Stats()(df)

    assert isinstance(stats, State)


def test_grouped_simple():
    con_df = df[["con_1", "con_2", "name"]]
    con_stats = Grouped(ContinuousStats(), "name")(con_df)

    a = 5
