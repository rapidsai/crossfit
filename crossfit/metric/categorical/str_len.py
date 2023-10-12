from crossfit.metric.continuous.mean import Mean


class MeanStrLength(Mean):
    @classmethod
    def from_array(self, array, *, axis: int) -> "MeanStrLength":
        return Mean(count=len(array), sum=array.str.len().sum(axis=axis))
