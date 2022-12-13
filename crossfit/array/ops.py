import functools

import numpy as np
from dask.utils import Dispatch


class NPBackendDispatch(Dispatch):
    def __call__(self, np_func, arg, *args, **kwargs):
        backend = self.dispatch(type(arg))

        return backend(np_func, arg, *args, **kwargs)


np_backend_dispatch = NPBackendDispatch(name="np_backend_dispatch")


class NPFunctionDispatch(Dispatch):
    def __init__(self, function, name=None):
        super().__init__(name=name)
        self.function = function

    def __call__(self, arg, *args, **kwargs):
        if isinstance(arg, np.ndarray):
            return self.function(arg, *args, **kwargs)

        if self.supports(arg):
            return super().__call__(arg, *args, **kwargs)

        return np_backend_dispatch(self.function, arg, *args, **kwargs)

    def supports(self, arg) -> bool:
        try:
            self.dispatch(type(arg))

            return True
        except TypeError:
            return False


def with_dispatch(func):
    dispatch = NPFunctionDispatch(func, name=func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return dispatch(*args, **kwargs)

    wrapper.dispatch = func

    return wrapper


class NPBackend:
    def __init__(self, np_like_module):
        self.np = np_like_module

    def __call__(self, np_func, *args, **kwargs):
        fn_name = np_func.__name__
        if hasattr(self, fn_name) and callable(getattr(self, fn_name)):
            fn = getattr(self, fn_name)
        else:
            fn = getattr(self.np, fn_name)

        return fn(*args, **kwargs)


minimum = with_dispatch(np.minimum)
maximum = with_dispatch(np.maximum)
sum = with_dispatch(np.sum)
all = with_dispatch(np.all)
unique = with_dispatch(np.unique)


__all__ = ["maximum", "minimum", "all", "sum"]
