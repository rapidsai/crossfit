import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        path = item.location[0]
        if "/tf_backend/" in path:
            item.add_marker(pytest.mark.tensorflow)
        if "/torch_backend/" in path:
            item.add_marker(pytest.mark.pytorch)
