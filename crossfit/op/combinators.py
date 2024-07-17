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

from crossfit.op.base import Op


class Sequential(Op):
    def __init__(self, *ops, pre=None, cols=False, repartition=None, keep_cols=None):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.ops = ops
        self.repartition = repartition

        if keep_cols:
            for op in self.ops:
                op.keep_cols.extend(keep_cols)

    def call_dask(self, data):
        for op in self.ops:
            if self.repartition is not None:
                data = data.repartition(npartitions=self.repartition)

            data = op(data)

        return data

    def call(self, data):
        for op in self.ops:
            data = op(data)

        return data
