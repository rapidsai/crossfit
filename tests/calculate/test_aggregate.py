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

import crossfit as cf


def test_pre():
    def pre(x):
        return x + 1

    class Test(cf.Aggregator):
        def prepare(self, data):
            return {"test": data + 1}

    agg = cf.Aggregator({"inc": pre}, pre=pre)
    agg_extended = Test({"inc": pre}, pre=pre)

    assert agg.prepare(0)["inc"] == 2
    assert agg_extended.prepare(0)["test"] == 2
