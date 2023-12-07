# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import functools
import inspect
import sys
import types
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional, Set, TypeVar, Union

import astunparse
import numpy as np

from crossfit.array import np_backend_dispatch
from crossfit.array import numpy as cnp

_CALL_HANDLER_ID = "__crossfit_call_handler__"
_CLOSURE_WRAPPER_ID = "__crossfit_closure_wrapper__"
_EMPTY_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
)
_CNP_IMPORT = ast.ImportFrom(
    module="crossfit.array.dispatch",
    names=[ast.alias(name="cnp", asname="cnp")],
    level=0,
)


FuncType = TypeVar("FuncType", bound=types.FunctionType)


def crossnp(func: FuncType, with_cache=True, validate_array_type=None) -> FuncType:
    """Make `func` work with various backends that implement the numpy-API.

    A few different scenarios are supported:
    1. Pass in a numpy function and get back the corresponding function from cnp
    2. A custom function that uses numpy functions.


    Parameters
    __________
    func: Callable
        The function to make work with various backends.
    with_cache: bool, optional
        Whether to cache the compiled function. Default is True.
    validate_array_type: Optional[Type], optional
        The type of array to validate the input arguments against. Default is None.


    Returns
    _______
    Callable
        The function that works with various backends.

    """

    # Check if func is a numpy function
    if isinstance(func, np.ufunc) or func.__module__ == "numpy":
        return getattr(cnp, func.__name__)

    cross_func = _compiler(func, with_cache=with_cache, validate_array_type=validate_array_type)

    if func == cross_func:
        func.__np__ = func
        func.__crossnp__ = cross_func

        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], np.ndarray):
            return func(*args, **kwargs)

        return cross_func(*args, **kwargs)

    wrapper.__np__ = func
    wrapper.__crossnp__ = cross_func

    return wrapper


class _AstModule(ast.NodeTransformer):
    """
    An AST visitor that extracts information about a module from the AST.

    Parameters
    ----------
    module_name : str
        The name of the module to extract information about.
    node : Optional[ast.AST], optional
        The AST to visit. If provided, the visitor will extract information about the module
        from this AST. Default is None.
    """

    def __init__(self, to_check: str, module: types.ModuleType, node: Optional[ast.AST] = None):
        self.to_check = to_check
        self.module = module
        self.aliases: Set[str] = set()
        self.imported: Set[str] = set()

        if node is not None:
            self.visit(node)

    def __contains__(self, node: ast.Call) -> bool:
        """Check if the provided node is a call to the module.

        Parameters
        ----------
        node : ast.Call
            The node to check.

        Returns
        -------
        bool
            True if the node is a call to the module, False otherwise.
        """
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in self.aliases:
                return True

        if isinstance(node.func, ast.Name):
            if node.func.id in self.imported:
                return True

        return False

    def __call__(self, node):
        """Visit an AST and extract information about the module.

        Parameters
        ----------
        node : ast.AST
            The AST to visit.

        Returns
        -------
        ast.AST
            The visited AST.
        """
        return self.visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Visit an import from node and extract information about
        the module if the import is from the module.

        Parameters
        ----------
        node : ast.ImportFrom
            The import from node to visit.

        Returns
        -------
        ast.ImportFrom
            The visited import from node.
        """
        if not getattr(node, "module", None):
            # This handles the case for relative imports
            module_name = ".".join(self.module.__name__.split(".")[: -node.level])
        else:
            module_name = node.module

        if module_name.startswith(self.to_check):
            self.imported.update(set(n.name for n in node.names))

        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Visit an import from node and extract information about
        the module if the import is from the module.

        Parameters
        ----------
        node : ast.Import
            The import from node to visit.

        Returns
        -------
        ast.Import
            The visited import from node.
        """
        for alias in node.names:
            if alias.name.startswith(self.to_check):
                if alias.asname is not None and alias.asname != alias.name:
                    self.aliases.add(alias.asname)

        return node


class _CrossNPAstTransformer(ast.NodeTransformer):
    """Ast NodeTransformer that replaces numpy calls with crossfit calls.

    Parameters
    ----------
    py_module : types.ModuleType
        The python module to transform.
    output : ast.ImportFrom, optional
        The import statement to use for crossfit. Default is an import of the `array` module
        from `crossfit` as `cnp`.
    """

    def __init__(
        self,
        py_module: types.ModuleType,
        output=_CNP_IMPORT,
    ):
        self.py_module = py_module
        self.file_ast = ast.parse(open(inspect.getsourcefile(py_module)).read())
        self.numpy_module = _AstModule("numpy", self.py_module, node=self.file_ast)
        self.output = output
        self.output_name = getattr(output.names[0], "asname", None) or output.names[0].name

    def __call__(
        self, node_or_fn: Union[ast.AST, types.FunctionType], validate_array_type=None
    ) -> Optional[ast.AST]:
        """
        Transform a node or function in the module.

        Parameters
        ----------
        node_or_fn : Union[ast.AST, types.FunctionType]
            The node or function to transform.
        validate_array_type: Optional[Type], optional
            The type of array to validate the input arguments against. Default is None.

        Returns
        -------
        Optional[ast.AST]
            The transformed node, or None if no transformation was performed.
        """
        if isinstance(node_or_fn, types.FunctionType):
            node = ast.parse(inspect.getsource(node_or_fn))
        else:
            node = node_or_fn

        _validate_temp = getattr(self, "_validate_array_type", None)
        self._validate_array_type = validate_array_type

        orig = deepcopy(node)
        output = self.visit(node)

        self._validate_array_type = _validate_temp

        if not _compare_ast(orig, output):
            output.body[0].name = _cross_np_name(output.body[0].name, self.py_module.__name__)
            output.body[0].body.insert(0, self.output)
            output = ast.fix_missing_locations(output)

            return output

        return None

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """
        Visit a call node and transform it if it is a numpy call.

        Parameters
        ----------
        node : ast.Call
            The call node to visit.

        Returns
        -------
        ast.Call
            The transformed call node.
        """
        if node in self.numpy_module:
            if isinstance(node.func, ast.Attribute):
                # TODO: Make this dynamic
                if not hasattr(cnp, node.func.attr):
                    raise ValueError(f"{node.func.attr} is not a valid attribute")
                node.func.value.id = self.output_name
                fn_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                fn_name = node.func.id
                if not hasattr(cnp, fn_name):
                    raise ValueError(f"{node.func.attr} is not a valid attribute")
                node.func = ast.Attribute(
                    value=ast.Name(id=self.output_name, ctx=ast.Load()),
                    attr=fn_name,
                    ctx=ast.Load(),
                )

            if self._validate_array_type:
                if not isinstance(self._validate_array_type, (list, tuple)):
                    self._validate_array_type = [self._validate_array_type]

                for array_type in self._validate_array_type:
                    backend = np_backend_dispatch.get_backend(array_type)

                    if fn_name not in backend:
                        # TODO: Show how to add the function to the backend
                        framework = array_type.__module__.split(".")[0]
                        raise ValueError(f"Function {fn_name} is not supported by {framework}")
        else:
            if isinstance(node.func, ast.Name):
                _maybe_transformed = self._maybe_compile(node.func.id)

                if _maybe_transformed is not None:
                    node.func.id = _maybe_transformed.__name__

            elif isinstance(node.func, ast.Attribute):
                if not isinstance(node.func.value, ast.Name):
                    # TODO: What to do here?
                    print(astunparse.unparse(node))

                    return node
                module = getattr(self.py_module, node.func.value.id, None)
                if not module:
                    print(astunparse.unparse(node))

                    return node
                _maybe_transformed = self._maybe_compile(node.func.attr, module=module)

                if _maybe_transformed is not None:
                    node.func = ast.Name(id=_maybe_transformed.__name__, ctx=ast.Load())

        return node

    def generic_visit(self, node):
        # Extra check to make sure we don't get errors for c-extension modules
        if not hasattr(node, "_fields"):
            return node

        return super().generic_visit(node)

    def _maybe_compile(
        self,
        node_name: str,
        module: Optional[types.ModuleType] = None,
        with_cache: bool = True,
    ) -> Optional[types.FunctionType]:
        if module is None:
            module = self.py_module
        if not hasattr(module, node_name):
            return None
        fn = getattr(module, node_name)
        compiled = _compiler(
            fn,
            validate_array_type=self._validate_array_type,
            with_cache=with_cache,
        )

        if fn != compiled:
            return compiled

        return None


class _CrossNPCompiler:
    """
    A compiler that can transform numpy calls to crossfit calls in functions.

    Parameters
    ----------
    keep_ast : bool, optional
        Whether to keep the AST for each transformed function. Default is False.
    """

    def __init__(self, keep_ast=True):
        self.modules = {}
        self.non_np = set()
        self.keep_ast = keep_ast
        self.ignore = set(sys.builtin_module_names)
        self.compiled = {}
        self.ignore.update(["contextlib", "warnings", "numpy"])
        if keep_ast:
            self.fn_to_ast = {}

    def __call__(
        self, fn: types.FunctionType, with_cache=True, validate_array_type=None
    ) -> types.FunctionType:
        """
        Transform a function by replacing numpy calls with crossfit calls (if it contains any).

        Parameters
        ----------
        fn : types.FunctionType
            The function to transform.
        with_cache: bool, optional
            Whether to cache the compiled function. Default is True.
        validate_array_type: Optional[Type], optional
            The type of array to validate the input arguments against. Default is None.

        Returns
        -------
        types.FunctionType
            The transformed function if it contains numpy calls, otherwise the original function.
        """

        if fn in self.non_np and with_cache:
            return fn

        cross_np_fn_name = _cross_np_fn_name(fn)
        if cross_np_fn_name in self.compiled and with_cache:
            return self.compiled[cross_np_fn_name]

        if fn.__module__ not in self.modules:
            module = inspect.getmodule(fn)
            if not module or module.__name__ in self.ignore:
                return fn
            try:
                fn_transformer = _CrossNPAstTransformer(module)
                self.modules[fn.__module__] = fn_transformer
            except Exception:
                return fn
        else:
            fn_transformer = self.modules[fn.__module__]

        maybe_transformed = fn_transformer(fn, validate_array_type=validate_array_type)
        if maybe_transformed is not None:
            if self.keep_ast:
                self.fn_to_ast[cross_np_fn_name] = maybe_transformed

            cross_fn = self.to_crossnp_fn(fn, maybe_transformed)
            if with_cache:
                self.compiled[cross_np_fn_name] = cross_fn

            return cross_fn

        self.non_np.add(fn)

        # TODO: Should we throw an exception here?
        return fn

    def to_crossnp_fn(self, fn: types.FunctionType, ast_node: ast.AST) -> types.FunctionType:
        """
        Compile a transformed node to a function.

        Parameters
        ----------
        fn : types.FunctionType
            The function to transform.
        ast_node : ast.AST
            The transformed node to compile.

        Returns
        -------
        types.FunctionType
            The compiled function.
        """
        write_to_file(ast_node, fn)

        code = compile(ast_node, inspect.getsourcefile(fn), "exec")

        ast_node = _wrap_ast_for_fn_with_closure_vars(ast_node, fn)
        # Compile the modified AST, and then find the function code object within
        # the returned module-level code object.
        code = compile(ast_node, inspect.getsourcefile(fn), "exec")
        code = _unwrap_code_for_fn(code, fn)

        closure = list(fn.__closure__ or ())
        closure = tuple(closure)

        fn_globals = {**fn.__globals__, **self.compiled}

        # Then, create a function from the compiled function code object, providing
        # the globals and the original function's closure.
        crossnp_fn = types.FunctionType(code, fn_globals, closure=closure)
        crossnp_fn.__defaults__ = fn.__defaults__
        crossnp_fn.__kwdefaults__ = fn.__kwdefaults__

        self.compiled[_cross_np_fn_name(fn)] = crossnp_fn

        return crossnp_fn

    def __contains__(self, fn) -> bool:
        """Check if a function has been transformed.

        Parameters
        __________
        fn : types.FunctionType
            The function to check.


        Returns
        _______
        bool
            True if the function has been transformed, False otherwise.
        """
        return _cross_np_name(fn, module_name=fn.__module__) in globals()


_compiler = _CrossNPCompiler(keep_ast=True)


def _cross_np_name(orig_name: str, module_name: Optional[str] = None) -> str:
    """
    Generate a name by combining a module name and an original name.

    Parameters
    ----------
    orig_name : str
        The original name to be combined with the module name.
    module_name : str, optional
        The name of the module. If provided, it will be combined with the original name to
        generate the resulting name. Default is None.

    Returns
    -------
    str
        The resulting name.
    """
    name_parts = ["__crossnp_", f"{orig_name}_"]
    if module_name:
        name_parts.append(module_name.replace(".", "_"))

    return "_".join(name_parts)


def _cross_np_fn_name(fn: types.FunctionType) -> str:
    """
    Generate a name for a function by combining its module name and original name.

    Parameters
    ----------
    fn : function
        The function for which to generate the name.

    Returns
    -------
    str
        The resulting name.
    """
    return _cross_np_name(fn.__name__, module_name=fn.__module__)


def _compare_ast(
    node1: Union[ast.expr, List[ast.expr]], node2: Union[ast.expr, List[ast.expr]]
) -> bool:
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in {"lineno", "end_lineno", "col_offset", "end_col_offset", "ctx"}:
                continue
            if not _compare_ast(v, getattr(node2, k)):
                return False
        return True

    elif isinstance(node1, list) and isinstance(node2, list):
        return all(_compare_ast(n1, n2) for n1, n2 in zip_longest(node1, node2))
    else:
        return node1 == node2


def _wrap_ast_for_fn_with_closure_vars(
    module: ast.Module,
    fn: types.FunctionType,
) -> ast.Module:
    """Wraps `module.body` in a function that defines closure variables for `fn`.
    If `fn` has any free variables (i.e., it's `__code__.co_freevars` is not
    empty), we want to make sure that compiling its AST (assumed to be in the body
    of `module`) will create the same set of free variables in the resulting code
    object. However, by default this won't happen, since we would be compiling
    `fn`'s AST in the absence of its original context (e.g., just compiling a
    nested function, and not the containing one).
    To work around this issue, this function wraps `module.body` in another
    `FunctionDef` that defines dummy variables corresponding to `fn`'s free
    variables. This causes the subsequent compile step to create the right set of
    free variables, and allows us to use `fn.__closure__` when creating a
    new function object via `types.FunctionType`.
    We also add <_CALL_HANDLER_ID> as a final dummy variable, and append its value
    (the call handler) to `fn.__closure__` when creating the new function object.
    Effectively, this wrapping looks like the following Python code:
        def __auto_config_closure_wrapper__():
            closure_var_1 = None
            closure_var_2 = None
            ...
            <_CALL_HANDLER_ID> = None
            def fn(...):  # Or some expression involving a lambda.
            ...  # Contains references to the closure variables.
    Args:
        module: An `ast.Module` object whose body contains the function definition
        for `fn` (e.g., as an `ast.FunctionDef` or `ast.Lambda`).
        fn: The function to create dummy closure variables for (assumed to
        correspond to the body of `module`).
    Returns:
        A new `ast.Module` containing an additional wrapper `ast.FunctionDef` that
        defines dummy closure variables.
    """
    ast_name = lambda name: ast.Name(id=name, ctx=ast.Store())  # noqa: E731
    ast_none = ast.Constant(value=None)
    closure_var_definitions = [
        ast.Assign(targets=[ast_name(var_name)], value=ast_none)
        for var_name in fn.__code__.co_freevars + (_CALL_HANDLER_ID,)
    ]

    wrapper_module = ast.Module(
        body=[
            ast.FunctionDef(
                name=_CLOSURE_WRAPPER_ID,
                args=_EMPTY_ARGUMENTS,
                body=[
                    *closure_var_definitions,
                    *module.body,
                ],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    wrapper_module = ast.fix_missing_locations(wrapper_module)

    return wrapper_module


def _find_function_code(code: types.CodeType, fn_name: str):
    """Finds the code object within `code` corresponding to `fn_name`."""
    code = [const for const in code.co_consts if inspect.iscode(const) and const.co_name == fn_name]
    assert len(code) == 1, f"Couldn't find function code for {fn_name!r}."
    return code[0]


def _unwrap_code_for_fn(code: types.CodeType, fn: types.FunctionType):
    """Unwraps `code` to find the code object for `fn`.
    This function assumes `code` is the result of compiling an `ast.Module`
    returned by `_wrap_node_for_fn_with_closure_vars`.
    Args:
        code: A code object containing code for `fn`.
        fn: The function to find a code object for within `code`.
    Returns:
        The code object corresponding to `fn`.
    """
    code = _find_function_code(code, _CLOSURE_WRAPPER_ID)
    code = _find_function_code(code, _cross_np_fn_name(fn))
    return code


def _make_closure_cell(contents):
    """Returns `types.CellType(contents)`."""
    if hasattr(types, "CellType"):
        # `types.CellType` added in Python 3.8.
        return types.CellType(contents)  # pytype: disable=wrong-arg-count
    else:
        # For earlier versions of Python, build a dummy function to get CellType.
        dummy_fn = lambda: contents  # noqa: E731
        cell_type = type(dummy_fn.__closure__[0])
        return cell_type(contents)


def write_to_file(ast_node, fn):
    path = Path(".crossfit")
    path.mkdir(exist_ok=True)

    # file_path = path / f"{module}.py"
    file_path = path / "fns.py"

    lines = astunparse.unparse(ast_node).splitlines()
    lines.insert(4, f"    from {fn.__module__} import *")
    with open(file_path, "a") as f:
        for line in lines:
            f.write(line + "\n")
    # with open(file_path, "w") as f:
    #     f.write(f"from {fn.__module__} import *")
    #     f.write(astunparse.unparse(ast_node))
