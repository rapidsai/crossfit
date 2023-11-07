import pytest


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        path = item.location[0]
        if "/tf_backend/" in path:
            item.add_marker(pytest.mark.tensorflow)
        if "/torch_backend/" in path:
            item.add_marker(pytest.mark.pytorch)
        if "/jax_backend/" in path:
            item.add_marker(pytest.mark.jax)
        if "/test_sklearn.py" in path:
            item.add_marker(pytest.mark.tensorflow)
            item.add_marker(pytest.mark.pytorch)
            item.add_marker(pytest.mark.jax)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
