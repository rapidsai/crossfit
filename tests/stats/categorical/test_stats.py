import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.categorical.stats import CategoricalStats

df = pd.DataFrame(
    {
        "a": ["a", "b", "c", "d", "e"] * 2,
        "a2": ["f", "g", "h", "i", "j"] * 2,
        "b": ["foo", "bar"] * 5,
        "c": ["hello", "world"] * 5,
    }
)


def test_categorical_stats_per_col():
    mf: MetricFrame = calculate_per_col(df, CategoricalStats())
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, pd.DataFrame)
