from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.common import CommonStats
from tests.utils import sample_df


@sample_df(dict(col=[1] * 5 + [None] * 5))
def test_common_basic_test(df):

    mf = calculate_per_col(CommonStats(), df)
    assert isinstance(mf, MetricFrame)

    result = mf.result().iloc[0]
    assert isinstance(result, df._constructor_sliced)

    assert result["count"] == 10
    assert result["num_missing"] == 5
