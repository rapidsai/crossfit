from __future__ import annotations

from typing import Callable, List

from crossfit.dataframe.core import CrossFrame
from crossfit.dataframe.dispatch import frame_dispatch


class PandasDataFrame(CrossFrame):
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

    @classmethod
    def concat(
        cls,
        frames: List[CrossFrame],
        ignore_index: bool = False,
        axis: int = 0,
    ):
        return frame_dispatch(
            cls._lib().concat(
                [frame.data for frame in frames],
                ignore_index=ignore_index,
                axis=axis,
            )
        )

    @classmethod
    def from_dict(cls, data: dict, index=None):
        df = cls._lib().DataFrame()
        for k, v in data.items():
            if hasattr(v, "shape"):
                # Make sure scalars are reshaped
                df[k] = v if v.shape else v.reshape((1,))
            else:
                df[k] = v
        if index is not None:
            df = df.set_index(cls._lib().Index(index))
        return frame_dispatch(df)

    def to_dict(self):
        # Clear index information
        _df = self.data.set_index(self._lib().RangeIndex(len(self.data)))
        return {col: _df[col] for col in self.columns}

    @property
    def columns(self):
        return list(self.data.columns)

    def select_column(self, column: str | int):
        return self.data[column]

    def project_columns(self, columns: list | tuple | str | int):
        if isinstance(columns, (int, str)):
            columns = [columns]  # Make sure we get a DataFrame
        return frame_dispatch(self.data[columns])

    def groupby_apply(self, by: list, func: Callable):
        grouped = self.data.groupby(by)
        result = grouped.apply(func)
        result.index = self._lib().Index(grouped.groups)
        return frame_dispatch(result)

    def groupby_partition(self, by: list) -> dict:
        grouped = self.data.groupby(by)
        return {
            slice_key: frame_dispatch(grouped.obj.loc[slice])
            for slice_key, slice in dict(grouped.groups).items()
        }

    def pivot(self, index=None, columns=None, values=None):
        return frame_dispatch(
            self.data.pivot(index=index, columns=columns, values=values)
        )


@frame_dispatch.register_lazy("numpy")
def register_numpy_backend():
    try:
        import pandas as pd
        import numpy as np

        @frame_dispatch.register(np.ndarray)
        def _numpy_to_pandas(data, index=None, column_name="data"):
            return PandasDataFrame(pd.DataFrame({column_name: data}, index=index))

    except ImportError:
        pass


@frame_dispatch.register_lazy("pandas")
def register_pandas_backend():
    import pandas as pd

    @frame_dispatch.register(pd.DataFrame)
    def _pd_frame(data):
        return PandasDataFrame(data)

    @frame_dispatch.register(pd.Series)
    def _pd_series(data, index=None, column_name="data"):
        if index is None:
            index = data.index
        return PandasDataFrame(pd.DataFrame({column_name: data}, index=index))
