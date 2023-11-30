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

from crossfit.metric.base import CrossMetric, state


class CommonStats(CrossMetric):
    count = state(init=0, combine=sum)
    num_missing = state(init=0, combine=sum)

    def __init__(self, count=None, num_missing=None):
        self.setup(count=count, num_missing=num_missing)

    def prepare(self, array) -> "CommonStats":
        return CommonStats(count=len(array), num_missing=len(array[array.isnull()]))

    def present(self):
        return {"count": self.count, "num_missing": self.num_missing}
