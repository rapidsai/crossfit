import numpy as np

import crossfit.array as cnp


def test_convert_no_op():
    array = np.array([1, 2, 3])

    assert cnp.all(array == cnp.convert(array, np.ndarray))
