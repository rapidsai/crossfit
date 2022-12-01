import logging
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


# ================== cupy ==================
@dispatch_to_dlpack.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_to_dlpack.register(cp.ndarray)
    def cupy_to_dlpack(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to DLPack")
        try:
            return input_array.to_dlpack()
        except AttributeError:
            return input_array.toDlpack()


@dispatch_from_dlpack.register_lazy("cupy")
def register_cupy_from_dlpack():
    import cupy as cp

    @dispatch_from_dlpack.register_lazy(cp.ndarray)
    def cupy_from_dlpack(capsule) -> cp.ndarray:
        logging.debug(f"Converting {capsule} to cp.ndarray")
        try:
            return cp.from_dlpack(capsule)
        except AttributeError:
            return cp.fromDlpack(capsule)


@dispatch_to_array.register_lazy("cupy")
def register_cupy_to_array():
    import cupy as cp

    @dispatch_to_array.register(cp.ndarray)
    def cupy_to_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return cp.asnumpy(input_array)


@dispatch_from_array.register_lazy("cupy")
@dispatch_from_cuda_array.register_lazy("cupy")
def register_cupy_to_array():
    import cupy as cp

    @dispatch_from_array.register(cp.ndarray)
    @dispatch_from_cuda_array.register(cp.ndarray)
    def cupy_from_array(array) -> cp.ndarray:
        logging.debug(f"Converting {array} to cp.ndarray")
        return cp.asarray(array)


@dispatch_to_cuda_array.register_lazy("cupy")
def register_cudf_to_cuda_array():
    import cupy as cp

    @dispatch_to_cuda_array.register(cp.ndarray)
    def cudf_to_cuda_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to cp.ndarray")
        return input_array


# ================== cudf ==================
@dispatch_to_dlpack.register_lazy("cudf")
def register_cudf_to_dlpack():
    import cudf

    @dispatch_to_dlpack.register(cudf.Series)
    def cudf_to_dlpack(input_array: cudf.Series):
        logging.debug(f"Converting {input_array} to DLPack")
        return input_array.to_dlpack()


@dispatch_from_dlpack.register_lazy("cudf")
def register_cudf_from_dlpack():
    import cudf

    @dispatch_from_dlpack.register(cudf.Series)
    def cudf_from_dlpack(capsule) -> cudf.Series:
        logging.debug(f"Converting {capsule} to cudf.Series")
        return cudf.io.from_dlpack(capsule)


@dispatch_to_array.register_lazy("cudf")
def register_cudf_to_array():
    import cudf

    @dispatch_to_array.register(cudf.Series)
    def cudf_to_array(input_array: cudf.Series):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return input_array.to_numpy()


@dispatch_from_array.register_lazy("cudf")
@dispatch_from_cuda_array.register_lazy("cudf")
def register_cudf_from_dlpack():
    import cudf

    @dispatch_from_array.register(cudf.Series)
    @dispatch_from_cuda_array.register(cudf.Series)
    def cudf_from_array(array) -> cudf.Series:
        logging.debug(f"Converting {array} to cudf.Series")
        return cudf.Series(array)


# ================== tf ==================
@dispatch_to_dlpack.register_lazy("tensorflow")
def register_tf_to_dlpack():
    import tensorflow as tf

    @dispatch_to_dlpack.register(tf.Tensor)
    def tf_to_dlpack(input_array: tf.Tensor):
        logging.debug(f"Converting {input_array} to DLPack")
        return tf.experimental.dlpack.to_dlpack(input_array)


@dispatch_from_dlpack.register_lazy("tensorflow")
def register_tf_to_dlpack():
    import tensorflow as tf

    @dispatch_from_dlpack.register(tf.Tensor)
    def tf_from_dlpack(capsule) -> tf.Tensor:
        logging.debug(f"Converting {capsule} to tf.Tensor")
        return tf.experimental.dlpack.from_dlpack(capsule)


@dispatch_to_array.register_lazy("tensorflow")
def register_tf_to_dlpack():
    import tensorflow as tf

    @dispatch_to_array.register(tf.Tensor)
    def tf_to_array(input_array: tf.Tensor):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return input_array.numpy()


@dispatch_from_array.register_lazy("tensorflow")
def register_tf_to_dlpack():
    import tensorflow as tf

    @dispatch_from_array.register(tf.Tensor)
    def tf_from_array(array) -> tf.Tensor:
        logging.debug(f"Converting {array} to tf.Tensor")
        return tf.convert_to_tensor(array)


# ================== cupy ==================
@dispatch_to_dlpack.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_to_dlpack.register(cp.ndarray)
    def cupy_to_dlpack(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to DLPack")
        try:
            return input_array.to_dlpack()
        except AttributeError:
            return input_array.toDlpack()


@dispatch_from_dlpack.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_from_dlpack.register(cp.ndarray)
    def cupy_from_dlpack(capsule) -> cp.ndarray:
        logging.debug(f"Converting {capsule} to cp.ndarray")
        try:
            return cp.from_dlpack(capsule)
        except AttributeError:
            return cp.fromDlpack(capsule)


@dispatch_to_array.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_to_array.register(cp.ndarray)
    def cupy_to_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return cp.asnumpy(input_array)


@dispatch_from_array.register_lazy("cupy")
@dispatch_from_cuda_array.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_from_array.register(cp.ndarray)
    @dispatch_from_cuda_array.register(cp.ndarray)
    def cupy_from_array(array) -> cp.ndarray:
        logging.debug(f"Converting {array} to cp.ndarray")
        return cp.asarray(array)


@dispatch_to_cuda_array.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @dispatch_to_cuda_array.register(cp.ndarray)
    def cudf_to_cuda_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to cp.ndarray")
        return input_array


def convert(input: Any, to: Type[ToType]) -> ToType:
    if isinstance(input, to):
        return input

    # 1. Try through cuda-array
    try:
        return dispatch_from_cuda_array(
            dispatch_to_cuda_array(input), to
        )
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
