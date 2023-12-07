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

import abc
import functools as ft

import numpy as np

from crossfit.calculate.aggregate import Aggregator
from crossfit.calculate.module import CrossModule, state
from crossfit.data import crossarray


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

    def __call__(self, data, *args, **kwargs):
        with crossarray:
            return self + self.prepare(data, *args, **kwargs)


class CrossAxisMetric(CrossMetric, abc.ABC):
    def __init__(self, axis: int, **kwargs):
        self.axis = axis
        self.setup(**kwargs)


min_state = ft.partial(state, init=np.iinfo(np.int32).min, combine=np.minimum)
max_state = ft.partial(state, init=np.iinfo(np.int32).max, combine=np.maximum)


__all__ = ["CrossMetric", "CrossAxisMetric", "state", "min_state", "max_state"]
