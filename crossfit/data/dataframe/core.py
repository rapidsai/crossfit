from __future__ import annotations

from typing import Callable, List

from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.dispatch import CrossFrame


class FrameBackend:
    def __init__(self, data):
        self.__data = data

    @property
    def data(self):
        """Wrapped frame-like object"""
        return self.__data

    def __setitem__(self, key, val):
        raise SyntaxError(
            f"In-place column assignment not supported. "
            f"Please use: new = old.assign({key}=value)"
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

    @classmethod
    def concat(
        cls,
        frames: List[FrameBackend],
        ignore_index: bool = False,
        axis: int = 0,
    ):
        """concatenate a list of ``CrossFrame`` obects

        Must return a new ``CrossFrame`` instance.
        """
        return ArrayBundle(frames, ignore_index=ignore_index, axis=axis)

    # Abstract Methods
    # Sub-classes must define these methods

    def __len__(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict):
        """Convert a dict to a new ``CrossFrame`` object"""
        raise NotImplementedError()

    def to_dict(self, orient: str = "dict"):
        """Convert an CrossFrame to a dict"""
        raise NotImplementedError()

    @property
    def columns(self):
        """Return list of column names"""
        raise NotImplementedError()

    def assign(self, **kwargs):
        """Set the value for a column

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def column(self, column: str | int):
        """Select a single column as an Array

        Must return an array-like object
        """
        raise NotImplementedError()

    def project(self, columns: list | tuple | str | int):
        """Select a column or list of columns

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def groupby_partition(self, by: list):
        """Partition an CrossFrame by group

        Must return a dictionary of new ``CrossFrame`` instances.
        """
        raise NotImplementedError()

    def apply(self, func: Callable, columns: list or None = None, **kwargs):
        """Apply a function to all data

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def groupby_apply(self, by: list, func: Callable, columns: list or None = None):
        """Execute a groupby-apply operation

        NOTE: This method is not yet used, but should be faster
        than looping over the result of ``groupby_partition``

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def pivot(self, index=None, columns=None, values=None):
        """Return reshaped CrossFrame

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def set_index(self, index):
        """Set the index of the CrossFrame

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()


# Make sure frame_dispatch(FrameBackend) -> FrameBackend
@CrossFrame.register(CrossFrame)
def _(data):
    return data


# Fall-back `ArrayBundle` definition
class ArrayBundle(FrameBackend):
    def __len__(self):
        if not hasattr(self, "_len"):
            _len = None
            for k, v in self.data.items():
                if _len is None:
                    _len = len(v)
                elif len(v) != _len:
                    raise ValueError(
                        f"Column {k} was length {len(v)}, but "
                        f"expected length {_len}"
                    )
            self._len = _len
        return self._len

    @classmethod
    def concat(
        cls,
        frames: List[FrameBackend],
        ignore_index: bool = False,
        axis: int = 0,
    ):
        from crossfit.data.array.ops import concatenate

        assert len(frames)
        assert not ignore_index  # TODO: Handle this
        assert axis == 0

        columns = frames[0].columns
        for frame in frames:
            assert columns == frame.columns

        combined = {
            column: concatenate([frame.column(column) for frame in frames])
            for column in columns
        }
        return ArrayBundle(combined)

    @classmethod
    def from_dict(cls, data: dict):
        return ArrayBundle(data)

    def to_dict(self):
        return self.data

    @property
    def columns(self):
        return list(self.data.keys())

    def assign(self, **kwargs):
        data = self.data.copy()
        for k, v in kwargs.items():
            if self.columns and len(v) != len(self):
                raise ValueError(
                    f"Column {k} was length {len(v)}, but "
                    f"expected length {len(self)}"
                )
        data.update(**kwargs)
        return CrossFrame(data)

    def column(self, column: str | int):
        return self.data[column]

    def project(self, columns: list | tuple | str | int):
        if isinstance(columns, (int, str)):
            columns = [columns]
        if not set(columns).issubset(set(self.columns)):
            raise ValueError(f"Invalid projection: {columns}")
        return CrossFrame({k: v for k, v in self.data.items() if k in columns})

    def apply(self, func: Callable, columns: list or None = None, **kwargs):
        with crossarray:
            if columns is None:
                columns = self.columns
            return CrossFrame(
                {k: func(v, **kwargs) for k, v in self.data.items() if k in columns}
            )

    def groupby_apply(self, by: list, func: Callable):
        raise NotImplementedError()

    def groupby_partition(self, by: list) -> dict:
        raise NotImplementedError()

    def pivot(self, index=None, columns=None, values=None):
        raise NotImplementedError()

    def set_index(self, index):
        raise NotImplementedError()


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
    backends = set()
    for v in data.values():
        backends.add(type(v).__module__.split(".")[0])
    if len(backends) == 1:
        return CrossFrame(next(iter(data.values()))).from_dict(data)
    return ArrayBundle(data)


# Map ArrayBundle to ArrayBundle
@CrossFrame.register(ArrayBundle)
def _ab_frame(data):
    return data
