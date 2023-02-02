import pytest

import numpy as np
import jax_backend
import jax.numpy as jnp

from crossfit.data import convert_array


@pytest.mark.parametrize("array", [jnp.asarray, np.array])
def test_simple_convert(array):
    assert jax_backend.Array in convert_array.supports
    tensor = convert_array(array([1, 2, 3]), jax_backend.Array)

    assert isinstance(tensor, jax_backend.Array)
