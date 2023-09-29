from dask.utils import Dispatch


class _CrossFrameDispatch(Dispatch):
    def __call__(self, data, *args, **kwargs):
        # TODO: Fix this

        from crossfit.backend.cudf.dataframe import CudfDataFrame
        from crossfit.backend.pandas.dataframe import PandasDataFrame
        from crossfit.backend.dask.dataframe import DaskDataFrame

        backends = [PandasDataFrame, DaskDataFrame, CudfDataFrame]

        for backend in backends:
            if isinstance(data, getattr(backend._lib(), "DataFrame")):
                return backend(data, *args, **kwargs)

        return super().__call__(data, *args, **kwargs)


CrossFrame = _CrossFrameDispatch(name="frame_dispatch")
