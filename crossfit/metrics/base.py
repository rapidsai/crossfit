import abc

from crossfit.calculate.module import CrossModule, state
from crossfit.data import crossarray
from crossfit.calculate.aggregate import Aggregator


class CrossMetric(CrossModule, abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    def to_aggregator(metric, pre=None, post=None) -> Aggregator:
        class MetricAggregator(Aggregator):
            def prepare(self, *args, **kwargs):
                return metric(*args, **kwargs)

            def present(self, state):
                return state.result

        return MetricAggregator(pre=pre, post=post)

    @property
    def result(self):
        if self._setup:
            return self.present()

        return None

    def __call__(self, *args, **kwargs):
        with crossarray:
            return self + self.prepare(*args, **kwargs)


__all__ = ["CrossMetric", "state"]
