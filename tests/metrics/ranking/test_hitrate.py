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

import pytest

pytest.importorskip("cupy")

from crossfit.data.sparse.ranking import SparseBinaryLabels, SparseRankings  # noqa: E402
from crossfit.metric.ranking import HitRate  # noqa: E402

y1 = [0, 5]
y2 = [8, 9]
y3 = []
y4 = [1, 2, 3, 4, 5, 6]
y5 = [3]
y6 = [0, 1]
y7 = [[]]

r1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r2 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
r3 = []
r4 = [1, 6, 8]

yn1 = [0, 3, 4]
yn2 = [2, 0, 5]

rn1 = [0, 1, 2]
rn2 = [2, 1, 0]


class TestHitrate:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y5, r1, [0.0]),
            (3, y5, r2, [0.0]),
            (4, y5, r1, [1.0]),
            (4, y5, r2, [0.0]),
            (4, y1, r1, [float("nan")]),
            (4, y1, r2, [float("nan")]),
            (10, y3, r3, [float("nan")]),
            (10, y5, r3, [0.0]),
            (10, y3, r2, [float("nan")]),
            (5, y5, [r1, r2], [1.0, 0.0]),
            (5, [y5, y1], [r1, r2], [1.0, float("nan")]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = HitRate(k).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)
