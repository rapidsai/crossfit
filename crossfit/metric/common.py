from crossfit.metric.base import CrossMetric, state


class CommonStats(CrossMetric):
    count = state(init=0, combine=sum)
    num_missing = state(init=0, combine=sum)

    def __init__(self, count=None, num_missing=None):
        self.setup(count=count, num_missing=num_missing)

    def prepare(self, array) -> "CommonStats":
        return CommonStats(count=len(array), num_missing=len(array[array.isnull()]))

    def present(self):
        return {"count": self.count, "num_missing": self.num_missing}
