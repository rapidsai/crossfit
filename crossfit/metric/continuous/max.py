from crossfit.metric.base import CrossAxisMetric, max_state


class Max(CrossAxisMetric):
    result = max_state()

    def __init__(self, result=None, axis=0):
        super().__init__(axis=axis, result=result)

    def prepare(self, array):
        return Max(sum=array.sum(axis=self.axis))

    def present(self):
        return self.result
