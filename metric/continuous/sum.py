from crossfit.metric.base import CrossAxisMetric, state


class Sum(CrossAxisMetric):
    result = state(init=0, combine=sum)

    def __init__(self, result=None, axis=0):
        super().__init__(axis=axis, result=result)

    def prepare(self, array):
        return Sum(sum=array.sum(axis=self.axis))

    def present(self):
        return self.result
