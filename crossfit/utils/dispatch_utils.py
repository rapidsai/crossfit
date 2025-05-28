# Copyright 2025 NVIDIA CORPORATION
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

# Code recycled from Dask: https://github.com/dask/dask/blob/main/dask/utils.py
class Dispatch:
    """Simple single dispatch."""

    def __init__(self, name=None):
        self._lookup = {}
        self._lazy = {}
        if name:
            self.__name__ = name

    def register(self, type, func=None):
        """Register dispatch of `func` on arguments of type `type`"""

        def wrapper(func):
            if isinstance(type, tuple):
                for t in type:
                    self.register(t, func)
            else:
                self._lookup[type] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def register_lazy(self, toplevel, func=None):
        """
        Register a registration function which will be called if the
        *toplevel* module (e.g. 'pandas') is ever loaded.
        """

        def wrapper(func):
            self._lazy[toplevel] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls):
        """Return the function implementation for the given ``cls``"""
        lk = self._lookup
        for cls2 in cls.__mro__:
            # Is a lazy registration function present?
            toplevel, _, _ = cls2.__module__.partition(".")
            try:
                register = self._lazy[toplevel]
            except KeyError:
                pass
            else:
                register()
                self._lazy.pop(toplevel, None)
                return self.dispatch(cls)  # recurse
            try:
                impl = lk[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache lookup
                    lk[cls] = impl
                return impl
        raise TypeError(f"No dispatch for {cls}")

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of argument.
        """
        meth = self.dispatch(type(arg))
        return meth(arg, *args, **kwargs)

    @property
    def __doc__(self):
        try:
            func = self.dispatch(object)
            return func.__doc__
        except TypeError:
            return "Single Dispatch for %s" % self.__name__


def supports(dispatch: Dispatch) -> set:
    if dispatch._lazy:
        _to_remove = set()
        for module in dispatch._lazy:
            try:
                register = dispatch._lazy[module]
            except KeyError:
                pass
            else:
                try:
                    register()
                    _to_remove.add(module)
                except ModuleNotFoundError:
                    pass
        for module in _to_remove:
            dispatch._lazy.pop(module, None)

    return set(dispatch._lookup)
