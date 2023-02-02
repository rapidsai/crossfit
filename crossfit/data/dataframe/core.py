from __future__ import annotations

from typing import Callable, List

from crossfit.data.dataframe.dispatch import frame_dispatch


class CrossFrame:
    def __init__(self, data):
        self.__data = data

    @property
    def data(self):
        """Wrapped frame-like object"""
        return self.__data

    def __getitem__(self, key):
        raise AttributeError(
            "getitem indexing not supported for CrossFrame. "
            "Please use `project_columns` or `select_column`. "
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"<CrossFrame: data={self.data.__repr__()}>"

    # Abstract Methods
    # Sub-classes must define these methods

    @classmethod
    def concat(
        cls,
        frames: List[CrossFrame],
        ignore_index: bool = False,
        axis: int = 0,
    ):
        """concatenate a list of ``CrossFrame`` obects

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict, index=None):
        """Convert a dict to a new ``CrossFrame`` object"""
        raise NotImplementedError()

    def to_dict(self, orient: str = "dict"):
        """Convert an CrossFrame to a dict"""
        raise NotImplementedError()

    @property
    def columns(self):
        """Return list of column names"""
        raise NotImplementedError()

    def select_column(self, column: str | int):
        """Select a single column

        Must return an array-like object
        """
        raise NotImplementedError()

    def project_columns(self, columns: list | tuple | str | int):
        """Select a column or list of columns

        Must return a new ``CrossFrame`` instance.
        """
        raise NotImplementedError()

    def groupby_partition(self, by: list):
        """Partition an CrossFrame by group

        Must return a dictionary of new ``CrossFrame`` instances.
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


# Make sure frame_dispatch(CrossFrame) -> CrossFrame
@frame_dispatch.register(CrossFrame)
def _(data):
    return data
