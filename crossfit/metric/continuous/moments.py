# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
