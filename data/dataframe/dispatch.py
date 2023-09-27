from dask.utils import Dispatch


class _CrossFrameDispatch(Dispatch):
    def __call__(self, data):
        return super().__call__(data)


CrossFrame = _CrossFrameDispatch(name="frame_dispatch")
