from crossfit.metrics.base import CrossMetric, state


class CommonStats(CrossMetric):
    count = state(init=0, combine=sum)
    num_missing = state(init=0, combine=sum)

    def prepare(self, array) -> "CommonStats":
        return CommonStats(count=len(array), num_missing=len(array[array.isnull()]))
