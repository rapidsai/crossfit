import pytest

import numpy as np
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils._array_api import get_namespace

from crossfit.data import crossarray, np_backend_dispatch


arr1 = [1, 2, 3]
arr2 = [4, 5, 6]


tensor_types = [
    m
    for m in np_backend_dispatch.supports
    if not m.__module__.startswith(("cupy", "cudf"))
]


@pytest.mark.parametrize("array_type", tensor_types)
def test_crossnp_type_of_target(array_type):
    backend = np_backend_dispatch.get_backend(array_type)
    tot = crossarray(type_of_target)
    con = [0.1, 0.6]

    if backend.namespace() != np:
        tensor = backend.asarray(con)
        namespace, _ = get_namespace(tensor)
        assert isinstance(namespace._namespace, type(backend.namespace()))

    assert tot(backend.asarray(con)) == tot(con)


@pytest.mark.parametrize(
    "metric",
    [
        "mean_squared_error",
        "median_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
    ],
)
@pytest.mark.parametrize("array_type", tensor_types)
def test_crossnp_sklearn_regression(metric, array_type):
    metric = crossarray(getattr(metrics, metric))
    backend = np_backend_dispatch.get_backend(array_type)

    cnp_out = metric(
        backend.asarray(arr1, dtype=backend.float32),
        backend.asarray(arr2, dtype=backend.float32),
    )
    np_out = metric(np.array(arr1, dtype=np.float32), np.array(arr2, dtype=np.float32))

    assert cnp_out == np_out


@pytest.mark.parametrize(
    "metric",
    [metrics.accuracy_score, metrics.precision_score, metrics.recall_score],
)
@pytest.mark.parametrize("array_type", tensor_types)
def test_crossnp_sklearn_clf(metric, array_type):
    backend = np_backend_dispatch.get_backend(array_type)

    y_true = np.random.randint(2, size=1000)
    y_pred = np.random.rand(1000)

    cross = crossarray(metric)

    cnp_out = cross(backend.asarray(y_true), backend.asarray(y_pred) > 0.5)
    np_out = cross(y_true, y_pred > 0.5)

    assert cnp_out == np_out
