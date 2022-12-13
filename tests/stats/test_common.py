import pandas as pd

from crossfit.core.calculate import calculate_per_col
from crossfit.core.frame import MetricFrame
from crossfit.stats.common import CommonStats


def test_common_basic_test():
    df = pd.DataFrame(dict(col=[1] * 5 + [None] * 5))

    mf = calculate_per_col(CommonStats(), df)
    assert isinstance(mf, MetricFrame)

    result = mf.result().iloc[0]
    assert isinstance(result, pd.Series)

    assert result["count"] == 10
    assert result["num_missing"] == 5
