from dask.utils import Dispatch


class _CrossFrameDispatch(Dispatch):
<<<<<<< HEAD
    def __call__(self, data):
        return super().__call__(data)
=======
    def __call__(self, data, *args, **kwargs):
        from crossfit.data.dataframe.core import FrameBackend

        if isinstance(data, FrameBackend):
            return data

        # TODO: Fix this

        from crossfit.backend.cudf.dataframe import CudfDataFrame
        from crossfit.backend.pandas.dataframe import PandasDataFrame
        from crossfit.backend.dask.dataframe import DaskDataFrame

        backends = [PandasDataFrame, DaskDataFrame, CudfDataFrame]

        for backend in backends:
            if isinstance(data, getattr(backend._lib(), "DataFrame")):
                return backend(data, *args, **kwargs)

        return super().__call__(data, *args, **kwargs)
>>>>>>> ae26d71e43324c3f1921a758052cf090422b40b8


CrossFrame = _CrossFrameDispatch(name="frame_dispatch")
