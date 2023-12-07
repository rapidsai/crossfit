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

from __future__ import annotations

from typing import Callable, List

from crossfit.data.array.conversion import convert_array
from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame


class PandasDataFrame(FrameBackend):
    @classmethod
    def _lib(cls):
        """Private method to return the backend library

        Used to allow the cudf backend to inherit from the
        pandas backend with minimal code duplication.
        Should not be used outside of ``PandasDataFrame``
        and its sub-classes.
        """
        import pandas as pd

        return pd

    def __len__(self):
        return len(self.data)

    @property
    def dtypes(self) -> dict:
        return self.data.dtypes.to_dict()

    @classmethod
    def concat(cls, frames: List[FrameBackend], axis: int = 0):
        # Validate frames
        if not isinstance(frames, list):
            raise TypeError(f"Expected list, got {type(frames)}")
        if len(frames) == 0:
            raise TypeError(f"Expected non-empty list, got {frames}")
        for frame in frames:
            if type(frame) is not cls:
                raise TypeError(f"All frames should be type {cls}, got {type(frame)}")

        return cls(
            cls._lib().concat(
                [frame.data for frame in frames],
                ignore_index=True if axis == 0 else False,
                axis=axis,
            )
        )

    @classmethod
    def from_dict(cls, data: dict):
        df = cls._lib().DataFrame()

        def _ensure_ser(col):
            typ = cls._lib().Series
            if isinstance(col, typ):
                return col
            return convert_array(col, typ)

        for k, v in data.items():
            if hasattr(v, "shape"):
                # Make sure scalars are reshaped
                df[k] = _ensure_ser(v if v.shape else v.reshape((1,)))
            else:
                df[k] = _ensure_ser(v)
        return cls(df)

    def to_dict(self):
        # Clear index information
        _df = self.data.set_index(self._lib().RangeIndex(len(self.data)))
        return {col: _df[col] for col in self.columns}

    @property
    def columns(self):
        return list(self.data.columns)

    def assign(self, **kwargs):
        return self.__class__(self.data.assign(**kwargs))

    def column(self, column: str | int):
        return self.data[column]

    def project(self, columns: list | tuple | str | int):
        if isinstance(columns, (int, str)):
            columns = [columns]  # Make sure we get a DataFrame
        if not set(columns).issubset(set(self.columns)):
            raise ValueError(f"Invalid projection: {columns}")
        return self.__class__(self.data[columns])

    def apply(self, func: Callable, *args, **kwargs):
        return self.__class__(self.data.apply(func, *args, **kwargs))

    def take(self, indices, axis=0):
        return self.__class__(self.data.take(indices, axis=axis))

    def groupby_indices(self, by: list) -> dict:
        if isinstance(by, (str, int, tuple)):
            by = [by]
        _df = self.data.set_index(self._lib().RangeIndex(len(self.data)))
        return dict(_df.groupby(by).groups)


@CrossFrame.register_lazy("numpy")
def register_numpy_backend():
    try:
        import numpy as np
        import pandas as pd

        @CrossFrame.register(np.ndarray)
        def _numpy_to_pandas(data, name="data"):
            return PandasDataFrame(pd.DataFrame({name: data}))

    except ImportError:
        pass


@CrossFrame.register_lazy("pandas")
def register_pandas_backend():
    import pandas as pd

    @CrossFrame.register(pd.DataFrame)
    def _pd_frame(data):
        return PandasDataFrame(data)

    @CrossFrame.register(pd.Series)
    def _pd_series(data, name="data"):
        return PandasDataFrame(pd.DataFrame({name: data}, index=data.index))
