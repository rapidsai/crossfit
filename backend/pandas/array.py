import logging

import pandas as pd

from crossfit.data.array import conversion


@conversion.dispatch_to_array.register(pd.Series)
def pandas_to_array(input_array: pd.Series):
    logging.debug(f"Converting {input_array} to np.ndarray")
    return input_array.to_numpy()


@conversion.dispatch_from_array.register(pd.Series)
def pandas_from_array(array) -> pd.Series:
    logging.debug(f"Converting {array} to pd.Series")
    return pd.Series(array)
