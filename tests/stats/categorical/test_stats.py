from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.categorical.stats import CategoricalStats
from tests.utils import sample_df

data = {
    "a": ["a", "b", "c", "d", "e"] * 2,
    "a2": ["f", "g", "h", "i", "j"] * 2,
    "b": ["foo", "bar"] * 5,
    "c": ["hello", "world"] * 5,
}


@sample_df(data)
def test_categorical_stats_per_col(df):
    mf: MetricFrame = calculate_per_col(CategoricalStats(), df)
    assert isinstance(mf, MetricFrame)

    result = mf.result()
    assert isinstance(result, type(df))
