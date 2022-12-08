from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin
from mypy.types import Type, Instance
from mypy.nodes import TypeInfo
from crossfit.dataframe import Backend
from mypy.checker import TypeChecker


def plugin(version: str):
    return CrossFitPlugin


class CrossFitPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_function_hook(self, fullname: str):
        if fullname.startswith("crossfit.dataframe") and fullname.endswith(
            "df_backend"
        ):
            return df_backend_hook

        if fullname.startswith("crossfit.dataframe") and fullname.endswith(
            "test_function"
        ):
            return test_hook

        return default_function_hook


def default_function_hook(ctx: FunctionContext) -> Type:
    return ctx.default_return_type


def test_hook(ctx: FunctionContext) -> Type:
    print(ctx)
    r = ctx.default_return_type.items  # type: ignore

    return r[0]


def df_backend_hook(ctx: FunctionContext) -> Type:
    # PandasBackend, CudfBackend = ctx.default_return_type.items  # type: ignore
    print(dir(ctx))
    print(ctx.default_return_type.type)
    print(type(ctx.default_return_type))
    print(ctx.default_return_type.serialize())
    # PandasBackend, CudfBackend = ("Module(pandas_backend)", "Module(cudf_backend)")

    cudf_inputs = ["cudf", "gpu"]

    print(type(ctx.api))
    print(dir(ctx.api))
    info = ctx.api.lookup_type("crossfit.dataframe.DataFrame")
    print(info)

    if str(ctx.arg_types[0]) in cudf_inputs:
        return Instance(ctx.default_return_type.type)

    return Instance(ctx.default_return_type.type, [])
