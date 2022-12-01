from dask.utils import Dispatch

class ArrayOpDispatch(Dispatch):
    def __call__(self, array):
        return super().__call__(input)
    
    
min = ArrayOpDispatch("min")
max = ArrayOpDispatch("max")
std = ArrayOpDispatch("std")

