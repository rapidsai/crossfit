from crossfit.metric.base import CrossAxisMetric, min_state


class Min(CrossAxisMetric):
    result = min_state()

    def __init__(self, result=None, axis=0):
        super().__init__(axis=axis, result=result)

    def prepare(self, array):
        return Min(sum=array.sum(axis=self.axis))

    def present(self):
        return self.result
