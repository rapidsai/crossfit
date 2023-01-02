import logging

from crossfit.array import conversion
from crossfit.array.dispatch import np_backend_dispatch, NPBackend

try:
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp
    from tensorflow.python.ops.numpy_ops import np_config
    from crossfit.utils import tf_utils

    np_config.enable_numpy_behavior()

    class TFBackend(NPBackend):
        def __init__(self):
            super().__init__(tnp)

        def namespace(self):
            return self

        def asarray(self, a, dtype=None, copy=False):
            return self.np.asarray(a, dtype=dtype)

        def unique_inverse(self, x):
            return self.unique(x, return_inverse=True)

        def unique_counts(self, x):
            return self.unique(x, return_counts=True)

        def unique_values(self, x):
            return self.unique(x)

        def concat(self, arrays, *, axis=None):
            _arrays = []

            for arr in arrays:
                if arr.dtype == arrays[0].dtype:
                    _arrays.append(arr)
                else:
                    _arrays.append(tf.cast(arr, dtype=arrays[0].dtype))

            return tf.concat(_arrays, axis or 0)

        def median(
            self,
            input_tensor,
            axis=None,
            out=None,
            overwrite_input=False,
            keepdims=False,
        ):
            # Get the shape of the input tensor
            shape = tf.shape(input_tensor)
            # Flatten the input tensor along the specified axis
            if axis is not None:
                input_tensor = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
                shape = tf.shape(input_tensor)
            # Flatten the input tensor
            flattened = tf.reshape(input_tensor, [-1])
            # Sort the flattened tensor
            sorted_tensor = tf.sort(flattened)
            # Get the middle element of the sorted tensor
            middle_index = shape[0] // 2
            # Check if the shape is odd or even
            if shape[0] % 2 == 0:
                # If the shape is even, return the average of the two middle elements
                result = (
                    sorted_tensor[middle_index - 1] + sorted_tensor[middle_index]
                ) / 2
            else:
                # If the shape is odd, return the middle element
                result = sorted_tensor[middle_index]
            # If keepdims is True, add the dimensions back to the result tensor
            if keepdims and axis is not None:
                result = tf.expand_dims(result, axis=axis)
            # If out is not None, write the result to the out tensor
            if out is not None:
                out.assign(result)
                return out

            return result

        def unique(self, input_tensor, return_counts=False):
            # Get the unique values in the input tensor
            unique_tensor, indices = tf.unique(input_tensor)

            # If return_counts is True, return both the unique values and their counts
            if return_counts:
                # Count the number of occurrences of each unique value
                count_tensor = tf.bincount(indices)
                return unique_tensor, count_tensor

            return unique_tensor

        def apply_along_axis(self, func, axis, input_tensor):
            # Get the shape of the input tensor
            shape = tf.shape(input_tensor)
            # Create a range tensor along the specified axis
            range_tensor = tf.range(shape[axis])

            # Create a function to apply to each element along the specified axis
            def element_func(i):
                # Select the element along the specified axis
                element = tf.gather(input_tensor, i, axis=axis)
                # Apply the function to the element
                result = func(element)
                # Return the result
                return result

            # Use tf.vectorized_map to apply the element function to
            #   each element in the range tensor
            result_tensor = tf.vectorized_map(element_func, range_tensor)
            # Reshape the result tensor to match the shape of the input tensor
            result_tensor = tf.reshape(result_tensor, shape)

            return result_tensor

        def union1d(self, a, b):
            a = tf.convert_to_tensor(a)
            b = tf.convert_to_tensor(b)

            if a.dtype != b.dtype:
                b = tf.cast(b, a.dtype)

            return tf.unique(tf.concat([a, b], axis=0))[0]

        def setdiff1d(self, x, y, assume_unique=False):
            return tf.compat.v1.setdiff1d(x, y).out

        def bincount(x, weights=None, minlength=None):
            x = tf.convert_to_tensor(x)
            if weights is not None:
                weights = tf.convert_to_tensor(weights)

            if minlength is not None:
                minlength = tf.convert_to_tensor(minlength)
                x = tf.maximum(x, minlength)

            if weights is None:
                counts, _ = tf.unique_with_counts(x)
            else:
                counts, _ = tf.unique_with_counts(x, weights=weights)

            max_x = tf.reduce_max(x)
            if minlength is None:
                minlength = max_x + 1

            result = tf.zeros(minlength, dtype=tf.int64)
            result = result + tf.scatter_nd(counts[:, :1], counts[:, 1], result.shape)

            return result

    tf_backend = TFBackend()
    np_backend_dispatch.register(tf.Tensor)(tf_backend)

    # Monkey patch tf.Tensor to support the array API namespace

    def tf_array_namespace(self):
        return tf_backend

    def tf_astype(self, dtype):
        return tf.cast(self, tf_utils.parse_dtype(dtype))

    setattr(tf.Tensor, "__array_namespace__", tf_array_namespace)
    setattr(tf.Tensor, "astype", tf_astype)
    setattr(tf.Tensor, "tolist", lambda self: self.numpy().tolist())
    setattr(tf.Tensor, "any", lambda self: tf.experimental.numpy.any(self))

    eq = tf.Tensor.__eq__

    def tf_equals(self, other):
        # TODO: Do this only when inside MonkeyPatchNumpy
        if not self.dtype == other.dtype:
            other = tf.cast(other, self.dtype)
        return eq(self, other)

    setattr(tf.Tensor, "__eq__", tf_equals)

    not_eq = tf.Tensor.__ne__

    def tf_not_equals(self, other):
        # TODO: Do this only when inside MonkeyPatchNumpy
        if not self.dtype == other.dtype:
            other = tf.cast(other, self.dtype)
        return not_eq(self, other)

    setattr(tf.Tensor, "__ne__", tf_not_equals)

    # Monkey patch tf.DType to support the array API namespace

    @property
    def kind(self):
        return tf_utils.get_dtype_kind(self)

    setattr(tf.DType, "kind", kind)


except ImportError:
    tf_backend = None


@conversion.dispatch_to_dlpack.register_lazy("tensorflow")
def register_tf_to_dlpack():
    import tensorflow as tf

    @conversion.dispatch_to_dlpack.register(tf.Tensor)
    def tf_to_dlpack(input_array: tf.Tensor):
        logging.debug(f"Converting {input_array} to DLPack")
        return tf.experimental.dlpack.to_dlpack(input_array)


@conversion.dispatch_from_dlpack.register_lazy("tensorflow")
def register_tf_from_dlpack():
    import tensorflow as tf

    @conversion.dispatch_from_dlpack.register(tf.Tensor)
    def tf_from_dlpack(capsule) -> tf.Tensor:
        logging.debug(f"Converting {capsule} to tf.Tensor")
        return tf.experimental.dlpack.from_dlpack(capsule)


@conversion.dispatch_to_array.register_lazy("tensorflow")
def register_tf_to_array():
    import tensorflow as tf

    @conversion.dispatch_to_array.register(tf.Tensor)
    def tf_to_array(input_array: tf.Tensor):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return input_array.numpy()


@conversion.dispatch_from_array.register_lazy("tensorflow")
def register_tf_from_array():
    import tensorflow as tf

    @conversion.dispatch_from_array.register(tf.Tensor)
    def tf_from_array(array) -> tf.Tensor:
        logging.debug(f"Converting {array} to tf.Tensor")
        return tf.convert_to_tensor(array)
