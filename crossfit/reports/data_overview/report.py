from crossfit.calculate.aggregate import Aggregator
from crossfit.metrics.continuous.range import Range
from crossfit.metrics.common import CommonStats
from crossfit.metrics.continuous.moments import Moments


class ContinuousMetrics(Aggregator):
    def prepare(self, array, *, axis):
        return {
            "range": Range().prepare(array, axis=axis),
            "moments": Moments().prepare(array, axis=axis),
            "common_stats": CommonStats().prepare(array),
        }
