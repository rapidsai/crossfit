import pytest
import astunparse

import numpy as np


from crossfit.array import crossarray
from crossfit.array import decorator as dec
from crossfit.utils import test_utils


def max_test(x, y):
    return np.maximum(x, y)


def nesting_test(x, y):
    return test_utils.min_test(x, y) + max_test(x, y)


@pytest.mark.parametrize(
    "fn", [np.all, np.sum, np.mean, np.std, np.var, np.any, np.prod]
)
def test_simple_numpy_function_crossnp(fn):
    crossfn = crossarray(fn)

    x = np.array([1, 2, 3])

    _cross_out = crossfn(x)
    _np_out = fn(x)

    assert np.all(_cross_out == _np_out)


@pytest.mark.parametrize("fn", [np.minimum, np.maximum, max_test, test_utils.min_test])
def test_combine_numpy_function_crossnp(fn):
    crossfn = crossarray(fn)

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    _cross_out = crossfn(x, y)
    _np_out = fn(x, y)

    assert np.all(_cross_out == _np_out)


def test_crossmap_ast_transformer_max_source():
    _compiler = dec._CrossNPCompiler(keep_ast=True)

    _compiler(max_test)
    max_ast = _compiler.fn_to_ast[dec._cross_np_fn_name(max_test)]
    max_source = astunparse.unparse(max_ast)

    _assert_source = """

def __crossnp__max_test__tests_array_test_decorator(x, y):
    from crossfit.array.dispatch import cnp as cnp
    return cnp.maximum(x, y)
"""
    assert max_source == _assert_source


def test_crossmap_ast_transformer_min():
    _compiler = dec._CrossNPCompiler(keep_ast=True)
    _compiler(test_utils.min_test)
    min_ast = _compiler.fn_to_ast[dec._cross_np_fn_name(test_utils.min_test)]
    min_source = astunparse.unparse(min_ast)

    _assert_source = """

def __crossnp__min_test__crossfit_utils_test_utils(x, y):
    from crossfit.array.dispatch import cnp as cnp
    return cnp.minimum(x, y)
"""
    assert min_source == _assert_source


def test_crossmap_ast_transformer_nesting_source():
    _compiler = dec._CrossNPCompiler(keep_ast=True)
    _compiler(nesting_test)
    nesting_ast = _compiler.fn_to_ast[dec._cross_np_fn_name(nesting_test)]
    nesting_source = astunparse.unparse(nesting_ast)

    _assert_source = """

def __crossnp__nesting_test__tests_array_test_decorator(x, y):
    from crossfit.array.dispatch import cnp as cnp
    return (__crossnp__min_test__crossfit_utils_test_utils(x, y) + __crossnp__max_test__tests_array_test_decorator(x, y))
"""  # noqa: E501
    assert nesting_source == _assert_source
