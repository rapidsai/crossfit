from crossfit.metrics.base import CrossMetric, state


class Sum(CrossMetric):
    sum = state(init=0, combine=sum)

    def __init__(self, pre=None, **kwargs):
        self._pre = pre
        self.setup(**kwargs)

    def prepare(self, array, axis=0):
        return Sum(sum=array.sum(axis=axis))

    def present(self):
        return self.sum
