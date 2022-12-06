from typing import Any, Type, TypeVar

import numpy as np
from dask.utils import Dispatch

InputType = TypeVar("InputType")
IntermediateType = TypeVar("IntermediateType")
ToType = TypeVar("ToType")


class ToDispatch(Dispatch):
    def __call__(self, input: InputType) -> IntermediateType:
        return super().__call__(input)


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


def convert(input: Any, to: Type[ToType]) -> ToType:
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
