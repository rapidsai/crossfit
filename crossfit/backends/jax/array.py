import logging

import numpy as np

from crossfit.core.array import conversion
from crossfit.core.array.dispatch import np_backend_dispatch, NPBackend


try:
    import jax
    import jax.numpy as jnp
    from jax import dlpack

    class JaxBackend(NPBackend):
        def __init__(self):
            super().__init__(jnp)

    jax_backend = JaxBackend()
    np_backend_dispatch.register(jax.Array)(jax_backend)

    @conversion.dispatch_to_dlpack.register(jax.Array)
    def jax_to_dlpack(input_array: jax.Array):
        logging.debug(f"Converting {input_array} to DLPack")
        return dlpack.to_dlpack(input_array)

    @conversion.dispatch_from_dlpack.register(jax.Array)
    def jax_from_dlpack(capsule) -> jax.Array:
        logging.debug(f"Converting {capsule} to jax.Array")
        return dlpack.from_dlpack(capsule)

    @conversion.dispatch_to_array.register(jax.Array)
    def jax_to_array(input_array: jax.Array):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return np.asarray(input_array)

    @conversion.dispatch_from_array.register(jax.Array)
    def jax_from_array(array) -> jax.Array:
        logging.debug(f"Converting {array} to jax.Array")
        return jnp.array(array)

except ImportError:
    jax_backend = None
