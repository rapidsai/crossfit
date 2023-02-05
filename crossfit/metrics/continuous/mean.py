import functools as ft


from crossfit.metrics.base import CrossMetric, state


class Mean(CrossMetric):
    count = state(init=0, combine=sum)
    sum = state(init=0, combine=sum)

    def __init__(self, pre=None, **kwargs):
        self._pre = pre
        self.setup(**kwargs)

    def prepare(self, *args, axis=0, **kwargs):
        if self._pre:
            prepped = self._pre(*args, **kwargs)
            if isinstance(prepped, Mean):
                return prepped
            length = get_length(*args, **kwargs)

            return Mean(count=length, sum=prepped * length)

        return self.from_array(*args, axis=axis, **kwargs)

    @classmethod
    def from_array(self, array, *, axis: int) -> "Mean":
        return Mean(count=len(array), sum=array.sum(axis=axis))

    def present(self):
        return self.sum / self.count


def create_mean_metric(calculation):
    return ft.wraps(calculation)(Mean(pre=calculation))


def get_length(*args, **kwargs):
    return len(args[0])
