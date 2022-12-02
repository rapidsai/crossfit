import numpy as np
from dask.utils import Dispatch


class CombinerDispatch(Dispatch):
    def combiner(self, left, right):
        if not type(left) == type(right):
            raise TypeError(f"Cannot add {type(left)} and {type(right)}")

        return self(left, right)

    # def __repr__(self) -> str:
    #     return f"{Dispatch}({self.name})"


class _Sum:
    def combiner(self, left, right):
        return left + right


Min = CombinerDispatch("Min")
Max = CombinerDispatch("Max")
Sum = _Sum()


# ================== numpy ==================


@Min.register(np.ndarray)
def np_min(left, right) -> np.ndarray:
    return np.minimum(left, right)


@Max.register(np.ndarray)
def np_max(left, right) -> np.ndarray:
    return np.maximum(left, right)
