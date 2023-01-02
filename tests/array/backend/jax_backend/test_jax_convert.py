import pytest

import numpy as np
import jax
import jax.numpy as jnp

import crossfit.array as cnp


@pytest.mark.parametrize("array", [jnp.asarray, np.array])
def test_simple_convert(array):
    assert jax.Array in cnp.convert.supports
    tensor = cnp.convert(array([1, 2, 3]), jax.Array)

    assert isinstance(tensor, jax.Array)
