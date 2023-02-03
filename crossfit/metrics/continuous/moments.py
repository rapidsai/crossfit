import numpy as np

from crossfit.metrics.base import CrossMetric, state


class Moments(CrossMetric):
    count = state(init=0, combine=sum)
    mean = state(init=0, combine=np.mean)
    var = state(init=0, combine=np.var)

    def prepare(self, array, *, axis) -> "Moments":
        return Moments(
            count=len(array), mean=array.mean(axis=axis), var=array.var(axis=axis)
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

        return Moments(count=new_count, mean=new_mean, var=new_var)

    def present(self):
        return {"mean": self.mean, "var": self.var}
