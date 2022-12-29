import types
from typing import TypeVar
import functools

import numpy as np
from crossfit.array.dispatch import cnp, with_dispatch


class MonkeyPatchNumpy:
    no_dispatch = {"dtype", "errstate", "may_share_memory", "finfo", "ndarray"}

    @classmethod
    def np_patch_dict(cls, orig_np):
        to_update = {
            key: with_dispatch(val)
            # key: monkey_patch_numpy(key, orig_np)
            for key, val in orig_np.items()
            if (
                key not in cls.no_dispatch
                and not key.startswith("_")
                and isinstance(val, types.FunctionType)
            )
        }

        return to_update

    def __enter__(self):
        self._original_numpy = np.__dict__.copy()
        patch_dict = self.np_patch_dict(self._original_numpy)
        np.__dict__.update(patch_dict)
        np.__origdict__ = self._original_numpy

    def __exit__(self, *args):
        np.__dict__.clear()
        np.__dict__.update(self._original_numpy)


FuncType = TypeVar("FuncType", bound=types.FunctionType)


def crossnp(func: FuncType) -> FuncType:
    """Make `func` work with various backends that implement the numpy-API.

    A few different scenarios are supported:
    1. Pass in a numpy function and get back the corresponding function from cnp
    2. A custom function that uses numpy functions.


    Parameters
    __________
    func: Callable
        The function to make work with various backends.


    Returns
    _______
    Callable
        The function that works with various backends.

    """

    try:
        import sklearn

        sklearn.set_config(array_api_dispatch=True)
    except ImportError:
        pass

    if isinstance(func, np.ufunc) or func.__module__ == "numpy":
        return getattr(cnp, func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MonkeyPatchNumpy():
            return func(*args, **kwargs)

    return wrapper
