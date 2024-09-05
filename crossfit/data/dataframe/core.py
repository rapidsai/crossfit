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

from functools import cached_property
from typing import Callable, List

from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.dispatch import CrossFrame


class FrameBackend:
    """
    Abstract DataFrame-object Wrapper

    Do not use this class directly.  Instead, use ``CrossFrame`` to
    automatically dispatch to the appropriate ``FrameBackend`` sub-class.

    Parameters
    ----------
    data: dict or DataFrame
        Wrapped DataFrame-like object
    """

    __slots__ = ("__data",)

    def __init__(self, data):
        self.__data = data

    @property
    def data(self):
        """Wrapped frame-like object"""
        return self.__data

    @property
    def columns(self) -> list:
        """Return column names

        Returns
        -------
        result: list
            List of column names contained in the ``FrameBackend`` object
        """
        raise NotImplementedError()

    @property
    def dtypes(self) -> dict:
        """Return dictionary of data types for each column

        Returns
        -------
        result: dict
            Dictionary mapping column-names to data types
        """
        raise NotImplementedError()

    def aggregate(self, agg: Callable, to_frame=False, **kwargs):
        """Apply a crossfit Aggregator

        Parameters
        ----------
        agg : Aggregator
            The crossfit Aggregator to apply
        to_frame : bool, default False
            Whether to summarize the result as a Pandas DataFrame
        **kwargs :
            Key-word arguments to pass through to ``Aggregator.prepare``

        Returns
        -------
        result: Dict[str, CrossMetric] or pd.DataFrame
            Aggregation result. Will correspond to a Pandas DataFrame-based
            summary of metric results if ``to_frame=True``, otherwise this
            will be a dict of ``CrossMetric`` objects.
        """
        from crossfit.calculate.aggregate import Aggregator

        if not isinstance(agg, Aggregator):
            raise TypeError()

        result = agg.prepare(self, **kwargs)
        if to_frame:
            return agg.present(result, to_frame=True)
        return result

    def groupby_partition(self, by: list):
        """Partition a FrameBackend object by group

        Parameters
        ----------
        by : list
            List of columns to group by

        Returns
        -------
        result: Dict[tuple, FrameBackend]
            Dictionary of group keys and ``FrameBackend`` partitions
        """
        if isinstance(by, (str, int, tuple)):
            by = [by]
        return {key: self.take(indices) for key, indices in self.groupby_indices(by).items()}

    def cast(self, columns: type | dict | None = None, backend: type | bool = True):
        """Cast column types and/or frame backend

        Parameters
        ----------
        columns : type or dict, optional
            New Array type to cast columns to. A dictionary mapping
            from column name to array type may also be specified.
            Default is ``None`` (column-type casting will not occur).
        backend : bool or type, optional
            New ``FrameBackend`` sub-class to cast to. By default
            (``backend = True``), the type of the first column
            will  be used to infer the new target backend. If False,
            the ``FrameBackend`` sub-class will be preserved.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """

        import crossfit as cf

        # Deal with array casting
        frame = self
        if columns:
            if isinstance(columns, dict):
                frame = CrossFrame(self.to_dict())
                new_columns = {}
                for col, typ in columns.items():
                    if col not in frame.columns:
                        raise ValueError(f"{col} not in available columns: {frame.columns}")
                    try:
                        new_columns[col] = cf.convert_array(frame[col], typ)
                    except TypeError as err:
                        raise TypeError(
                            f"Unable to cast column {col} to {typ}.\nOriginal error: {err}"
                        )
                frame = frame.assign(**new_columns)
            else:
                try:
                    frame = CrossFrame(self.to_dict()).apply(cf.convert_array, columns)
                except TypeError as err:
                    raise TypeError(
                        f"Unable to cast all column types to {columns}.\nOriginal error: {err}"
                    )

        # Set backend if backend is True
        if backend is True and frame.columns:
            backend = CrossFrame(frame[frame.columns[0]]).__class__

        # Set backend to original class if backend is False
        if backend is False:
            backend = self.__class__

        # Cast backend and return
        if issubclass(backend, FrameBackend) and backend != frame.__class__:
            return backend.from_dict(frame.to_dict())

        if isinstance(backend, str):
            pass
        return frame

    @classmethod
    def concat(cls, frames: List[FrameBackend], axis: int = 0):
        """concatenate a list of FrameBackend objects

        Parameters
        ----------
        frames : List[FrameBackend]
            List of ``FrameBackend`` objects to concatenate.
        axis : int, default 0
            Axist to concatenate on.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict):
        """Convert a dict to a new FrameBackend object

        Parameters
        ----------
        data : Dict[str, Array]
            Dictionary of Array data. Dictionary keys will
            correspond to the column names in the output
            ``FrameBackend`` object.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Convert the FrameBackend object to a dict

        Returns
        -------
        result: dict
            Dictionary mapping column-names to Array data
        """
        raise NotImplementedError()

    def assign(self, **pairs):
        """Assign columns

        Parameters
        ----------
        **pairs : dict
            key-value pairs to assign. Keys correspond to
            column names, while pairs correspond to array-like objects.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """
        raise NotImplementedError()

    def column(self, column: str | int | tuple):
        """Select a single column

        Parameters
        ----------
        column : str, int or tuple
            Column name to select from the FrameBackend object.

        Returns
        -------
        result: Array-like object
            Array-like object corresponding to the selected column.
        """
        raise NotImplementedError()

    def project(self, columns: list | tuple | str | int):
        """Select a column or list of columns

        Parameters
        ----------
        columns : list, str, int or tuple
            Column names to select from the FrameBackend object.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object containing the selected columns.
        """
        raise NotImplementedError()

    def apply(self, func: Callable, *args, **kwargs):
        """Apply a function to all columns independently

        Parameters
        ----------
        func : Callable
            Function to apply to each column independently.
        *args :
            Positional arguments to pass through to ``func``.
        **kwargs :
            Key-word arguments to pass through to ``func``.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """
        raise NotImplementedError()

    def take(self, indices):
        """Return the elements in the given positional
        indices along an axis

        Parameters
        ----------
        indices : Array
            Popsitional indices to select from all columns.

        Returns
        -------
        result: FrameBackend
            New ``FrameBackend`` object
        """
        raise NotImplementedError()

    def groupby_indices(self, by: list) -> dict:
        """Partition a FrameBackend by group

        Parameters
        ----------
        by : list
            List of column names to group by.

        Returns
        -------
        result: Dict[tuple, Array]
            Dictionary mapping groupy keys to positional-index arrays.
        """
        raise NotImplementedError()

    def __setitem__(self, *args):
        raise SyntaxError(
            "In-place column assignment not supported. Please use: new = old.assign(key=value)"
        )

    def __getitem__(self, key):
        if isinstance(key, list):
            # List selection means column projection
            return self.project(key)
        elif isinstance(key, (int, str, tuple)):
            # Column selection should return an Array
            return self.column(key)
        else:
            raise KeyError(f"Unsupported getitem key: {key}")

    def __repr__(self):
        return f"<CrossFrame({self.__class__.__name__}): columns={self.columns}>"

    def __len__(self):
        raise NotImplementedError()


# Make sure frame_dispatch(FrameBackend) -> FrameBackend
@CrossFrame.register(CrossFrame)
def _(data):
    return data


# Fall-back `ArrayBundle` definition
class ArrayBundle(FrameBackend):
    @cached_property
    def _cached_len(self):
        _len = None
        for k, v in self.data.items():
            if _len is None:
                _len = len(v)
            elif len(v) != _len:
                raise ValueError(f"Column {k} was length {len(v)}, but expected length {_len}")
        return _len

    def __len__(self):
        return self._cached_len

    @property
    def dtypes(self) -> dict:
        # TODO: Does this work for "all" supported Array types?
        return {col: self[col].dtype for col in self.columns}

    @classmethod
    def concat(
        cls,
        frames: List[FrameBackend],
        axis: int = 0,
    ):
        from crossfit.data.array.ops import concatenate

        # Validate frames
        if not isinstance(frames, list):
            raise TypeError(f"Expected list, got {type(frames)}")
        if len(frames) == 0:
            raise TypeError(f"Expected non-empty list, got {frames}")

        if axis == 0:
            columns = frames[0].columns
            for frame in frames:
                if type(frame) is not cls:
                    raise TypeError(f"All frames should be type {cls}, got {type(frame)}")
                if columns != frame.columns:
                    raise TypeError("Cannot concatenat misaligned columns")

            combined = {
                column: concatenate([frame.column(column) for frame in frames])
                for column in columns
            }
        elif axis == 1:
            columns = set()
            combined = {}
            for frame in frames:
                if type(frame) is not cls:
                    raise TypeError(f"All frames should be type {cls}, got {type(frame)}")
                _columns = set(frame.columns)
                if _columns.intersection(columns):
                    intersection = _columns.intersection(columns)
                    raise TypeError(f"Concatenated columns intersect: {intersection}")
                columns |= _columns
                combined.update(frame.data)
        else:
            raise ValueError(f"axis={axis} not supported")

        return cls(combined)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data)

    def to_dict(self):
        return self.data

    @property
    def columns(self):
        return list(self.data.keys())

    def assign(self, **kwargs):
        data = self.data.copy()
        for k, v in kwargs.items():
            if self.columns and len(v) != len(self):
                raise ValueError(f"Column {k} was length {len(v)} but expected length {len(self)}")
        data.update(**kwargs)
        return self.__class__(data)

    def column(self, column: str | int):
        return self.data[column]

    def project(self, columns: list | tuple | str | int):
        if isinstance(columns, (int, str)):
            columns = [columns]
        if not set(columns).issubset(set(self.columns)):
            raise ValueError(f"Invalid projection: {columns}")
        return self.__class__({k: v for k, v in self.data.items() if k in columns})

    def apply(self, func: Callable, *args, **kwargs):
        with crossarray:
            data = {k: func(v, *args, **kwargs) for k, v in self.data.items()}
        return self.__class__(data)

    def take(self, indices, axis=0):
        import numpy as np

        assert axis == 0  # TODO: Support axis=1
        with crossarray:
            return self.__class__({k: np.take(v, indices, axis=axis) for k, v in self.data.items()})

    def groupby_indices(self, by: list) -> dict:
        if isinstance(by, (str, int, tuple)):
            by = [by]

        # Try projecting the grouped-on columns
        # to promote the CrossFrame to Pandas or cuDF
        if set(by) < set(self.columns):
            return self.project(by).cast().groupby_indices(by)

        # If simple projection doesn't work, try
        # casting all columns to Pandas Series'
        try:
            import pandas as pd

            return self.apply(convert_array, pd.Series).cast().groupby_indices(by)
        except ImportError:
            pass

        raise NotImplementedError("groupby_indices not implemented for ArrayBundle")

    def __repr__(self):
        name = self.__class__.__name__
        backends = ", ".join(set(str(d).split(".")[0] for d in self.dtypes.values()))

        return f"<CrossFrame({name}[{backends}]): columns={self.columns}>"


# Map Tensorflow data to ArrayBundle
@CrossFrame.register_lazy("tensorflow")
def register_tf_from_dlpack():
    import tensorflow as tf

    @CrossFrame.register(tf.Tensor)
    def _tf_to_bundle(data, name="data"):
        return ArrayBundle({name: data})


# Map PyTorch data to ArrayBundle
@CrossFrame.register_lazy("torch")
def register_torch_from_dlpack():
    import torch

    @CrossFrame.register(torch.Tensor)
    def _torch_to_bundle(data, name="data"):
        return ArrayBundle({name: data})


# Map dict to ArrayBundle
@CrossFrame.register(dict)
def _dict_frame(data):
    return ArrayBundle(data)


# Map ArrayBundle to ArrayBundle
@CrossFrame.register(ArrayBundle)
def _ab_frame(data):
    return data
