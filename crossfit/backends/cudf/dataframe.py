from crossfit.data.dataframe.dispatch import frame_dispatch
from crossfit.backends.pandas.dataframe import PandasDataFrame


class CudfDataFrame(PandasDataFrame):

    # Inherits most logic from PandasDataFrame, but
    # this space can be used to handle differences
    # in behavior when necessary

    @classmethod
    def _lib(cls):
        import cudf

        return cudf


@frame_dispatch.register_lazy("cupy")
def register_cupy_backend():
    try:
        import cudf
        import cupy

        @frame_dispatch.register(cupy.ndarray)
        def _cupy_to_cudf(data, index=None, column_name="data"):
            return CudfDataFrame(cudf.DataFrame({column_name: data}, index=index))

    except ImportError:
        pass


@frame_dispatch.register_lazy("cudf")
def register_cudf_backend():
    import cudf

    @frame_dispatch.register(cudf.DataFrame)
    def _cudf_dataframe(data):
        return CudfDataFrame(data)

    @frame_dispatch.register(cudf.Series)
    def _cudf_series(data, index=None, column_name="data"):
        if index is None:
            index = data.index
        return CudfDataFrame(cudf.DataFrame({column_name: data}, index=index))
