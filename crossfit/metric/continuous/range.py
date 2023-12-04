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


class Range(CrossAxisMetric):
    min = state(init=0, combine=np.minimum)
    max = state(init=0, combine=np.maximum)

    def prepare(self, array) -> "Range":
        return Range(axis=self.axis, min=array.min(axis=self.axis), max=array.max(axis=self.axis))

    def present(self):
        return {"min": self.min, "max": self.max}
