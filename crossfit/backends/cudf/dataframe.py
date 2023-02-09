from typing import Callable


from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.backends.pandas.dataframe import PandasDataFrame


class CudfDataFrame(PandasDataFrame):

    # Inherits most logic from PandasDataFrame, but
    # this space can be used to handle differences
    # in behavior when necessary

    @classmethod
    def _lib(cls):
        import cudf

        return cudf

    def apply(self, func: Callable, *args, **kwargs):
        try:
            return self.__class__(self.data.apply(func, *args, **kwargs))
        except ValueError:
            # Numba-compilation failed
            pass
        with crossarray:
            return CrossFrame(
                {k: func(v, *args, **kwargs) for k, v in self.data.items()}
            ).cast()


@CrossFrame.register_lazy("cupy")
def register_cupy_backend():
    try:
        import cudf
        import cupy

        @CrossFrame.register(cupy.ndarray)
        def _cupy_to_cudf(data, name="data"):
            return CudfDataFrame(cudf.DataFrame({name: data}))

    except ImportError:
        pass


@CrossFrame.register_lazy("cudf")
def register_cudf_backend():
    import cudf

    @CrossFrame.register(cudf.DataFrame)
    def _cudf_dataframe(data):
        return CudfDataFrame(data)

    @CrossFrame.register(cudf.Series)
    def _cudf_series(data, name="data"):
        return CudfDataFrame(cudf.DataFrame({name: data}, index=data.index))
