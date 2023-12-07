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

from crossfit.metric.base import CrossAxisMetric, state


class Sum(CrossAxisMetric):
    result = state(init=0, combine=sum)

    def __init__(self, result=None, axis=0):
        super().__init__(axis=axis, result=result)

    def prepare(self, array):
        return Sum(sum=array.sum(axis=self.axis))

    def present(self):
        return self.result
