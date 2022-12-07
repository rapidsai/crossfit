import pandas as pd
from pandas import *  # noqa: F401, F403


is_installed: bool = True


def is_grouped(df) -> bool:
    return isinstance(df, pd.core.groupby.generic.DataFrameGroupBy)
