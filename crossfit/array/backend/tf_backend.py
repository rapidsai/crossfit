import logging

from crossfit.array import conversion
from crossfit.array import ops


@ops.np_backend_dispatch.register_lazy("tensorflow")
def register_tf_backend():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    class TFBackend(ops.NPBackend):
        def __init__(self):
            super().__init__(tnp)

    ops.np_backend_dispatch.register(tf.Tensor)(TFBackend())


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
