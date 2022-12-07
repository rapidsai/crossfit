from crossfit.utils.df_utils import requires_df_backend

is_installed: bool = True

try:
    import cudf
    from cudf import *  # noqa: F401, F403
except ImportError:
    cudf = None
    is_installed = False


requires_cudf = requires_df_backend("cudf")


@requires_cudf
def is_grouped(df) -> bool:
    return isinstance(df, cudf.core.groupby.groupby.DataFrameGroupBy)
