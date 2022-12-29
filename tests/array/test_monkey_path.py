import numpy as np

from crossfit.array.monkey_patch import MonkeyPatchNumpy


def custom_function(a, b):
    # Custom function that uses numpy and returns
    # different results when monkey-patched

    c = np.add(a, b)

    if hasattr(np, "__origdict__"):
        c += 1

    return c


def test_monkey_path_np():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]

    from sklearn import metrics

    # Call the custom function within the context manager
    with MonkeyPatchNumpy():
        x = np.array(arr1)
        y = np.array(arr2)
        z = custom_function(x, y)
        met = metrics.mean_squared_error(x, y)

    assert not getattr(np, "__origdict__", None)
    assert np.all(met == metrics.mean_squared_error(arr1, arr2))
    assert np.all(z == custom_function(arr1, arr2) + 1)
