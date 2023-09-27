import numpy as np

from crossfit.metric.base import CrossAxisMetric, state


class Moments(CrossAxisMetric):
    count = state(init=0, combine=sum)
    mean = state(init=0, combine=np.mean)
    var = state(init=0, combine=np.var)

    def prepare(self, array) -> "Moments":
        return Moments(
            axis=self.axis,
            count=len(array),
            mean=array.mean(axis=self.axis),
            var=array.var(axis=self.axis),
        )

    def combine(self, other) -> "Moments":
        delta = other.mean - self.mean
        tot_count = self.count + other.count

        new_mean = self.mean + delta * other.count / tot_count
        m_self = self.var * max(self.count - 1, 1)
        m_other = other.var * max(other.count - 1, 1)
        M2 = m_self + m_other + (delta**2) * self.count * other.count / tot_count
        new_var = M2 / max(tot_count - 1, 1)
        new_count = tot_count

        return Moments(axis=self.axis, count=new_count, mean=new_mean, var=new_var)

    @property
    def std(self):
        return np.sqrt(self.var)

    def present(self):
        return {"mean": self.mean, "var": self.var, "std": self.std}
