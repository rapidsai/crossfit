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

import functools as ft

from crossfit.metric.base import CrossMetric, state


class Mean(CrossMetric):
    count = state(init=0, combine=sum)
    sum = state(init=0, combine=sum)

    def __init__(self, pre=None, **kwargs):
        self._pre = pre
        self.setup(**kwargs)

    def prepare(self, *args, axis=0, **kwargs):
        if self._pre:
            prepped = self._pre(*args, **kwargs)
            if isinstance(prepped, Mean):
                return prepped
            length = get_length(*args, **kwargs)

            return Mean(count=length, sum=prepped * length)

        return self.from_array(*args, axis=axis, **kwargs)

    @classmethod
    def from_array(self, array, *, axis: int) -> "Mean":
        return Mean(count=len(array), sum=array.sum(axis=axis))

    def present(self):
        return self.sum / self.count


def create_mean_metric(calculation):
    return ft.wraps(calculation)(Mean(pre=calculation))


def get_length(*args, **kwargs):
    return len(args[0])
