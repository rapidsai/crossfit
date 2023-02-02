import abc

from crossfit.core.module import CrossModule, state


class CrossMetric(CrossModule, abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def result(self):
        if self._setup:
            return self.present()

        return None

    def __call__(self, *args, **kwargs):
        return self + self.prepare(*args, **kwargs)
    
    
__all__ = [
    "CrossMetric",
    "state"
]