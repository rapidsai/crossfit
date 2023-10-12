from dask.utils import Dispatch


class _CrossFrameDispatch(Dispatch):
    def __call__(self, data, *args, **kwargs):
        from crossfit.data.dataframe.core import FrameBackend

        if isinstance(data, FrameBackend):
            return data

        # TODO: Fix this
        from crossfit.backend.pandas.dataframe import PandasDataFrame
        from crossfit.backend.dask.dataframe import DaskDataFrame
        
        backends = [PandasDataFrame, DaskDataFrame]
        

        try:
            from crossfit.backend.cudf.dataframe import CudfDataFrame
            backends.append(CudfDataFrame)
        except ImportError:
            pass    


        for backend in backends:
            if isinstance(data, getattr(backend._lib(), "DataFrame")):
                return backend(data, *args, **kwargs)

        return super().__call__(data, *args, **kwargs)


CrossFrame = _CrossFrameDispatch(name="frame_dispatch")
