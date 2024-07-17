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
pytest.importorskip("pytrec_eval")


import numpy as np  # noqa: E402
from pytrec_eval import RelevanceEvaluator  # noqa: E402

from crossfit.data.sparse.ranking import Rankings, SparseBinaryLabels, SparseRankings  # noqa: E402
from crossfit.metric.ranking import AP, Precision  # noqa: E402
from tests.pytrec_utils import create_qrel, create_results, create_run  # noqa: E402

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


class TestPrecision:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y2, r1, [0.0]),
            (10, y2, r1, [0.2]),
            (2, y2, r2, [1.0]),
            (10, y2, r3, [0.0]),
            (10, y3, r3, [float("nan")]),
            (10, y3, r2, [float("nan")]),
            (1, y2, r2, [1.0]),
            (3, y4, r4, [2.0 / 3]),
            (1, y1, [r1, r2], [1.0, 0.0]),
            (1, [y1, y2], [r1, r2], [1.0, 1.0]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = Precision(k).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)

    @pytest.mark.parametrize(
        "obs, scores",
        [
            # All-zero relevance
            (
                np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32),
                np.array(
                    [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
                    dtype=np.float32,
                ),
            ),
            # All-one relevance
            (
                np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int32),
                np.array(
                    [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
                    dtype=np.float32,
                ),
            ),
            # Non-continuous relevance
            (
                np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.int32),
                np.array(
                    [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
                    dtype=np.float32,
                ),
            ),
            # Empty inputs
            (np.array([[], []], dtype=np.int32), np.array([[], []], dtype=np.float32)),
            # Single-element lists
            (
                np.array([[1], [0]], dtype=np.int32),
                np.array([[0.1], [0.9]], dtype=np.float32),
            ),
        ],
    )
    def test_pytrec_eval(self, obs, scores):
        qrel = create_qrel(obs)
        run = create_run(scores)
        pytrec_result = RelevanceEvaluator(qrel, ["P.5"]).evaluate(run)

        scores = Rankings.from_scores(scores)
        labels = SparseBinaryLabels.from_matrix(obs)
        results = create_results({"P_5": Precision(5).score(labels, scores)})

        for query_id, metrics in results.items():
            for metric_name, value in metrics.items():
                assert value == pytest.approx(pytrec_result[query_id][metric_name], rel=1e-3)


class TestTruncatedPrecision:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y2, r1, [0.0]),
            (10, y2, r1, [0.2]),
            (2, y2, r2, [1.0]),
            (10, y2, r3, [float("nan")]),
            (10, y3, r3, [float("nan")]),
            (10, y3, r2, [float("nan")]),
            (1, y2, r2, [1.0]),
            (6, y4, r4, [2.0 / 3]),
            (6, y2, r4, [1.0 / 3]),
            (3, y2, r4, [1.0 / 3]),
            (1, y1, [r1, r2], [1.0, 0.0]),
            (1, [y1, y2], [r1, r2], [1.0, 1.0]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = Precision(k, truncated=True).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)


class TestAP:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y2, r1, [0.0]),
            (10, y2, r1, [0.155555556]),
            (2, y2, r2, [1.0]),
            (10, y2, r3, [0.0]),
            (5, y4, r4, [2.0 / 5]),
            (10, y3, r3, [float("nan")]),
            (10, y3, r2, [float("nan")]),
            (6, y1, [r1, r2], [1.333333333 / 2, 0.2 / 2]),
            (6, y1, [r1, r4, r1], [1.333333333 / 2, 0.0, 1.333333333 / 2]),
            (6, [y2, y3], [r2, r4], [1.0, float("nan")]),
            (6, [y5, y1], [r1, r2], [0.25, 0.2 / 2]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = AP(k).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)
