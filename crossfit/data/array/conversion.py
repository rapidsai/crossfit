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

from itertools import product
from typing import Any, Type, TypeVar

import numpy as np
from dask.utils import Dispatch

from crossfit.utils import dispatch_utils

InputType = TypeVar("InputType")
IntermediateType = TypeVar("IntermediateType")
ToType = TypeVar("ToType")


class ToDispatch(Dispatch):
    def __call__(self, input: InputType) -> IntermediateType:
        return super().__call__(input)

    def dispatch(self, cls):
        try:
            return super().dispatch(cls)
        except TypeError:
            for key, val in self._lookup.items():
                if issubclass(cls, key):
                    return val
            raise TypeError(f"Cannot convert {cls} to {self.name}")


class FromDispatch(Dispatch):
    def __call__(
        self,
        intermediate: IntermediateType,
        to: Type[ToType],
    ) -> ToType:
        meth = self.dispatch(to)
        return meth(intermediate)


dispatch_to_dlpack = ToDispatch("to_dlpack")
dispatch_from_dlpack = FromDispatch("from_dlpack")

dispatch_to_array = ToDispatch("to_array")
dispatch_from_array = FromDispatch("from_array")

dispatch_to_cuda_array = ToDispatch("to_cuda_array")
dispatch_from_cuda_array = FromDispatch("from_cuda_array")


# ================== numpy ==================
@dispatch_from_dlpack.register(np.ndarray)
def np_from_dlpack(capsule) -> np.ndarray:
    try:
        return np._from_dlpack(capsule)
    except AttributeError as exc:
        raise NotImplementedError(
            "NumPy does not implement the DLPack Standard until version 1.22.0, "
            f"currently running {np.__version__}"
        ) from exc


@dispatch_to_array.register(np.ndarray)
def np_to_array(input_array: np.ndarray):
    return input_array


@dispatch_from_array.register(np.ndarray)
@dispatch_from_cuda_array.register(np.ndarray)
def np_from_array(array) -> np.ndarray:
    return np.array(array)


class ArrayConverter:
    def __call__(self, input: Any, to: Type[ToType]) -> ToType:
        if isinstance(input, to):
            return input

        # 1. Try through cuda-array
        try:
            return dispatch_from_cuda_array(dispatch_to_cuda_array(input), to)
        except Exception:
            pass

        # 2. Try to DLPack
        try:
            return dispatch_from_dlpack(dispatch_to_dlpack(input), to)
        except Exception:
            pass

        # 3. Try through array
        try:
            return dispatch_from_array(dispatch_to_array(input), to)
        except Exception:
            pass

        # TODO: Check step here

        raise TypeError(
            f"Can't create {input} array from type {to}, "
            "which doesn't support any of the available conversion interfaces."
        )

    @property
    def supports(self):
        conversions = {}

        from_tos = [
            (dispatch_from_cuda_array, dispatch_to_cuda_array),
            (dispatch_from_dlpack, dispatch_to_dlpack),
            (dispatch_from_array, dispatch_to_array),
        ]

        for from_, to_ in from_tos:
            from_types = dispatch_utils.supports(from_)
            to_types = dispatch_utils.supports(to_)

            if from_types and to_types:
                types = [t for t in set(product(from_types, to_types)) if len(set(t)) > 1]

                for from_t, to_t in types:
                    if from_t in conversions:
                        conversions[from_t].add(to_t)
                    else:
                        conversions[from_t] = {to_t}

        return conversions


convert_array = ArrayConverter()
