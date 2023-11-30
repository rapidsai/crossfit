# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from crossfit.backend.pandas.dataframe import PandasDataFrame
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

    frame = CrossFrame(data).cast()
    assert isinstance(frame, PandasDataFrame)
    assert isinstance(CrossFrame(data).cast(backend=ArrayBundle), ArrayBundle)

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
    assert isinstance(frame.project(["b", "c"]).cast(), PandasDataFrame)
    assert isinstance(frame[["b", "c"]], ArrayBundle)

    # Test CrossFrame.apply
    frame2 = frame.concat([frame, frame])
    frame3 = frame2.assign(d=np.zeros(20))
    frame4 = frame3[["a", "d"]].apply(lambda x: x + 1)
    np.all(frame3["b"] == frame4["d"])

    # Test CrossFrame.groupby_partition.
    # will work with ArrayBundle, as long as
    # "grouped" columns can be promoted to
    # Pandas or cuDF
    partitions = frame3.groupby_partition("c")
    np.all(partitions["dog"]["c"] == "dog")
    np.all(partitions["cat"]["c"] == "cat")
    np.all(partitions["dog"]["a"] > partitions["cat"]["a"])
