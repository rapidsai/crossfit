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

from crossfit.data.sparse.ranking import (  # noqa: E402
    Rankings,
    SparseBinaryLabels,
    SparseNumericLabels,
    SparseRankings,
)
from crossfit.metric.ranking import DCG, NDCG  # noqa: E402
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


class TestDCG:
    @pytest.mark.parametrize(
        "y_gold,y_pred,expect,params",
        [
            (yn1, rn1, [5.616107762], {"log_base": "e"}),
            (yn1, rn2, [8.501497843], {"log_base": "e"}),
            (yn2, rn2, [6.0], {"log_base": "2"}),
            (yn1, r3, [0], {"log_base": "e"}),
            (y3, rn1, [float("nan")], {"log_base": "e"}),
            (y3, r3, [float("nan")], {"log_base": "e"}),
            (yn1, [rn1, rn2], [5.616107762, 8.501497843], {"log_base": "e"}),
            ([yn1, yn2], [rn1, rn2], [5.616107762, 8.656170245], {"log_base": "e"}),
        ],
    )
    def test_score(self, y_gold, y_pred, expect, params):
        y_gold = SparseNumericLabels.from_matrix(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = DCG(3, **params).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)


class TestNDCG:
    @pytest.mark.parametrize(
        "y_gold,y_pred,expect,params",
        [
            (yn1, rn1, [5.616107762 / 8.501497843], {}),
            (yn2, rn1, [6.492127684 / 9.033953658], {}),
            (yn2, rn2, [8.656170245 / 9.033953658], {}),
            (yn1, r3, [0], {}),
            (y3, rn1, [float("nan")], {}),
            (y3, r3, [float("nan")], {}),
            (yn1, [rn1, rn2], [5.616107762 / 8.501497843, 1.0], {}),
            (
                [yn1, yn2],
                [rn1, rn2],
                [5.616107762 / 8.501497843, 8.656170245 / 9.033953658],
                {},
            ),
        ],
    )
    def test_numeric_score(self, y_gold, y_pred, expect, params):
        y_gold = SparseNumericLabels.from_matrix(y_gold)
        if len(y_pred) == 0 or y_pred == r3 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = (
            NDCG(3, **params, log_base="e").score(y_gold, y_pred, nan_handling="propagate").tolist()
        )
        assert pred == pytest.approx(expect, nan_ok=True)

    @pytest.mark.parametrize(
        "y_gold,y_pred,expect,params",
        [
            (y1, r1, [1.956593383 / 2.352934268], {}),
            (y6, rn1, [1.0], {}),
            (y1, r4, [0.0], {}),
            (y6, rn2, [1.631586747 / 2.352934268], {}),
            (
                [y1, y6],
                [r1, rn2],
                [1.956593383 / 2.352934268, 1.631586747 / 2.352934268],
                {},
            ),
        ],
    )
    def test_binary_score(self, y_gold, y_pred, expect, params):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = (
            NDCG(10, **params, log_base="e")
            .score(y_gold, y_pred, nan_handling="propagate")
            .tolist()
        )

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
            # Non-binary relevance
            (
                np.array([[3, 2, 1, 0, 0], [0, 0, 1, 0, 2]], dtype=np.int32),
                np.array(
                    [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    def test_pytrec_eval(self, obs, scores):
        qrel = create_qrel(obs)
        run = create_run(scores)
        pytrec_result = RelevanceEvaluator(qrel, ["ndcg_cut.5"]).evaluate(run)

        scores = Rankings.from_scores(scores)
        labels = SparseNumericLabels.from_matrix(obs)
        results = create_results(
            {"ndcg_cut_5": NDCG(5).score(labels, scores, nan_handling="zerofill")}
        )

        for query_id, metrics in results.items():
            for metric_name, value in metrics.items():
                assert value == pytest.approx(pytrec_result[query_id][metric_name], rel=1e-3)
