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

import functools
import types
from typing import TypeVar

import numpy as np
from dask.utils import Dispatch

from crossfit.utils import dispatch_utils


class NPBackendDispatch(Dispatch):
    def __call__(self, np_func, arg, *args, **kwargs):
        jit = kwargs.pop("__jit", False)
        try:
            backend = self.dispatch(type(arg))
        except TypeError:
            return np_func(arg, *args, **kwargs)

        return backend(np_func, arg, *args, __jit=jit, **kwargs)

    def maybe_jit(self, fn, arg, jit=True):
        try:
            backend = self.dispatch(type(arg))
        except TypeError:
            return fn

        if hasattr(backend, "jit_compile"):
            return backend.jit_compile(fn, jit=jit)

        return fn

    def get_backend(self, array_type):
        return self.dispatch(array_type)

    @property
    def supports(self):
        return dispatch_utils.supports(self)


np_backend_dispatch = NPBackendDispatch(name="np_backend_dispatch")


class NPFunctionDispatch(Dispatch):
    def __init__(self, function, jit=False, name=None):
        super().__init__(name=name)
        self.function = function
        self.jit = jit

    def __call__(self, arg, *args, **kwargs):
        if self.function == np.dtype:
            return self.function(arg, *args, **kwargs)

        if isinstance(arg, (np.ndarray, list)):
            return self.function(arg, *args, **kwargs)

        if self.supports(arg):
            return super().__call__(arg, *args, **kwargs)

        return np_backend_dispatch(self.function, arg, *args, __jit=self.jit, **kwargs)

    def supports(self, arg) -> bool:
        try:
            self.dispatch(type(arg))

            return True
        except TypeError:
            return False


def with_dispatch(func, jit=False):
    dispatch = NPFunctionDispatch(func, name=func.__name__, jit=jit)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return dispatch(*args, **kwargs)

    wrapper.dispatch = func

    return wrapper


class ArrayBackend:
    """Class to provide a compatible interface for functions in a numpy-like module.

    This class is meant to be used with numpy-like modules (such as `cupy` or `numpyro`)
    and wraps functions to make them compatible with other numpy-like modules.

    Attributes
    ----------
    np : module
        A numpy-like module.

    """

    def __init__(
        self,
        np_like_module: types.ModuleType,
    ):
        """Initialize the class with a numpy-like module.

        Parameters
        ----------
        np_like_module : module
            A numpy-like module.

        """
        self.np = np_like_module

    def namespace(self) -> types.ModuleType:
        """Return the numpy-like module.

        Returns
        -------
        module
            The numpy-like module.

        """
        return np

    def __call__(self, np_func, *args, **kwargs):
        """Call the function with the given arguments.

        This method is used to call numpy-like functions.

        Parameters
        ----------
        np_func : callable
            The numpy-like function to be called.
        *args : tuple
            The arguments for the function.
        **kwargs : dict
            The keyword arguments for the function.

        Returns
        -------
        Any
            The result of calling the function.

        """
        jit = kwargs.pop("__jit", False)
        fn_name = np_func.__name__

        return self.get(fn_name, jit=jit)(*args, **kwargs)

    def __getattr__(self, name) -> types.FunctionType:
        """Get the attribute by name.

        If the attribute does not exist in the numpy-like module, a
        `NotImplementedError` is raised.

        Parameters
        ----------
        name : str
            The name of the attribute to get.

        Returns
        -------
        Any
            The attribute with the given name.

        Raises
        ------
        NotImplementedError
            If the attribute does not exist in the numpy-like module.

        """

        if not hasattr(self.np, name):
            raise NotImplementedError(f"Function {name} not implemented for {self.np}")
        fn = getattr(self.np, name)

        return fn

    def get(self, name, jit=False):
        del jit
        return getattr(self, name)

    def __contains__(self, np_func) -> bool:
        """Check if the numpy-like function exists in the backend.

        The method checks if the function exists in the backend and returns True if
        it exists.

        Parameters
        ----------
        np_func : str or callable
            The name or the function to check for existence.

        Returns
        -------
        bool
            True if the function exists in the backend, False otherwise.

        """
        if isinstance(np_func, str):
            fn_name = np_func
        else:
            fn_name = np_func.__name__

        # TODO: Clean this up
        if fn_name == "dtype":
            return True

        if fn_name in CrossArray.no_dispatch:
            return True

        if hasattr(self, fn_name) and callable(getattr(self, fn_name)):
            return True

        if hasattr(self.np, fn_name):
            return True

        return False


class CustomFunctionDispath(NPFunctionDispatch):
    def __call__(self, arg, *args, **kwargs):
        if self.supports(arg):
            return super().__call__(arg, *args, **kwargs)

        jit_kwargs = {}
        if isinstance(self.jit, dict):
            jit_kwargs = self.jit

        fn = np_backend_dispatch.maybe_jit(self.function, arg, **jit_kwargs)

        return fn(arg, *args, **kwargs)


class DispatchedNumpy:
    fns = {}
    no_dispatch = {"errstate", "may_share_memory", "finfo"}

    def __getattr__(self, name):
        return self.get(name)

    def get(self, name, jit_compile=False, overwrite=False):
        if name.startswith("__"):
            return super().__getattr__(name)

        if not hasattr(np, name):
            raise AttributeError("Unknown numpy function")

        np_fn = getattr(np, name)
        if name in self.no_dispatch:
            return np_fn

        if name not in self.fns or overwrite:
            self.fns[name] = with_dispatch(np_fn, jit=jit_compile)

        return self.fns[name]


numpy = DispatchedNumpy()
FuncType = TypeVar("FuncType", bound=types.FunctionType)


class CrossArray:
    """A context-manager that allows a function to work with various backends
    that implement the numpy API.
    """

    no_dispatch = {
        "dtype",
        "errstate",
        "may_share_memory",
        "finfo",
        "ndarray",
        "isscalar",
    }

    np_dict = np.__dict__.copy()
    stack = []

    def __init__(self):
        self.dispatch_dict = self.np_patch_dict(self.np_dict)

    @classmethod
    def np_patch_dict(cls, orig_np):
        """Generate a dictionary of numpy functions that are patched to work with various backends.

        Parameters
        ----------
        orig_np: dict
            The original `numpy` module's __dict__.

        Returns
        -------
        dict
            A dictionary of patched numpy functions.
        """

        to_update = {
            key: with_dispatch(val)
            for key, val in orig_np.items()
            if (
                key not in cls.no_dispatch
                and not key.startswith("_")
                and isinstance(val, types.FunctionType)
            )
        }

        return to_update

    def __enter__(self):
        """Enter the context-manager and patch numpy functions to work with various backends."""

        if not self.stack:
            np.__dict__.update(self.dispatch_dict)
            np.__origdict__ = self.np_dict
            self.stack = [True]
        else:
            self.stack.append(True)

    def __exit__(self, *args):
        """Exit the context-manager and restore the original numpy functions."""
        self.stack.pop()

        if not self.stack:
            np.__dict__.clear()
            np.__dict__.update(self.np_dict)

    def __call__(self, func: FuncType, jit=False, overwrite=False) -> FuncType:
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
        if isinstance(func, np.ufunc) or func.__module__ == "numpy":
            return numpy.get(func.__name__, jit_compile=jit, overwrite=overwrite)

        to_call = func
        if jit:
            to_call = CustomFunctionDispath(to_call)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return to_call(*args, **kwargs)

        return wrapper


crossarray = CrossArray()


__all__ = [
    "ArrayBackend",
    "with_dispatch",
    "np_backend_dispatch",
    "numpy",
    "crossarray",
]
