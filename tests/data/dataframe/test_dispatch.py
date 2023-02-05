import pytest

import numpy as np

from crossfit.backends.pandas.dataframe import PandasDataFrame
from crossfit.data.dataframe.core import ArrayBundle
from crossfit.data.dataframe.dispatch import CrossFrame


def test_pandas_frame():

    arr1 = np.arange(10)
    arr2 = np.ones(10)
    arr3 = np.array(["cat", "dog"] * 5)
    data = {
        "a": arr1,
        "b": arr2,
        "c": arr3,
    }

    frame = CrossFrame(data)
    assert isinstance(frame, PandasDataFrame)

    frame2 = frame.concat([frame, frame])
    frame3 = frame2.assign(d=np.zeros(20))
    frame4 = frame3[["a", "d"]].apply(lambda x: x + 1)

    np.all(frame3["b"] == frame4["d"])


def test_array_bundle():
    tf = pytest.importorskip("tensorflow")

    arr1 = tf.range(10)
    arr2 = np.ones(10)
    arr3 = np.array(["cat", "dog"] * 5)
    data = {
        "a": arr1,
        "b": arr2,
        "c": arr3,
    }

    # Mixed column backends will
    # produce a ArrayBundle
    frame = CrossFrame(data)
    assert isinstance(frame, ArrayBundle)

    # Projecting numpy-based columns will
    # produce a PandasDataFrame
    assert isinstance(frame[["b", "c"]], PandasDataFrame)

    frame2 = frame.concat([frame, frame])
    frame3 = frame2.assign(d=np.zeros(20))
    frame4 = frame3[["a", "d"]].apply(lambda x: x + 1)

    np.all(frame3["b"] == frame4["d"])
