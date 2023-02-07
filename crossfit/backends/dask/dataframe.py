from __future__ import annotations

from typing import Callable, List

from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame


@CrossFrame.register_lazy("dask")
def register_dask_backend():

    import dask.dataframe as dd

    class DaskDataFrame(FrameBackend):
        def compute(self, **kwargs):
            return CrossFrame(self.data.compute(**kwargs))

        def persist(self, **kwargs):
            return CrossFrame(self.data.persist(**kwargs))

        @classmethod
        def concat(
            cls,
            frames: List[FrameBackend],
            ignore_index: bool = False,
            axis: int = 0,
        ):
            return CrossFrame(
                dd.DataFrame.concat(frames, ignore_index=ignore_index, axis=axis)
            )

        def aggregate(self, agg, **kwargs):
            from crossfit.backends.dask.aggregate import aggregate as dask_aggregate

            return dask_aggregate(self.data, agg, **kwargs)

        def __len__(self):
            raise NotImplementedError("Please avoid calling `len` on Dask-based data.")

        @classmethod
        def from_dict(cls, *args, **kwargs):
            raise NotImplementedError()

        def to_dict(self, **kwargs):
            raise NotImplementedError()

        @property
        def columns(self):
            return list(self.data.columns)

        def assign(self, **kwargs):
            return CrossFrame(self.data.assign(**kwargs))

        def column(self, column: str | int):
            return self.data[column]

        def project(self, columns: list | tuple | str | int):
            if isinstance(columns, (int, str)):
                columns = [columns]  # Make sure we get a DataFrame
            if not set(columns).issubset(set(self.columns)):
                raise ValueError(f"Invalid projection: {columns}")
            return CrossFrame(self.data[columns])

        def apply(self, func: Callable, **kwargs):
            def _apply(x):
                return x.apply(func, **kwargs)

            return CrossFrame(self.data.map_partitions(_apply))

        def groupby_partition(self, *args, **kwargs):
            raise NotImplementedError()

    @CrossFrame.register(dd.DataFrame)
    def _dask_frame(data):
        return DaskDataFrame(data)
