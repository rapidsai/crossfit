import pytest

pytest.importorskip("cupy")

import numpy as np
from pytrec_eval import RelevanceEvaluator

from crossfit.data.sparse.ranking import (Rankings, SparseBinaryLabels,
                                          SparseRankings)
from crossfit.metric.ranking import Recall
from tests.pytrec_utils import create_qrel, create_results, create_run

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


class TestRecall:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,valid,expect",
        [
            (3, y2, r1, None, [0.0]),
            (10, y2, r1, [r for r in r1 if r not in [8]], [0.5]),
            (10, y2, r2, None, [1.0]),
        ],
    )
    def test_masked_score(self, k, y_gold, y_pred, valid, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        y_pred = SparseRankings.from_ranked_indices(y_pred, valid_items=valid)
        pred = Recall(k).score(y_gold, y_pred, nan_handling="propagate").tolist()

        assert pred == pytest.approx(expect, nan_ok=True)

    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y2, r1, [0.0]),
            (10, y2, r1, [1.0]),
            (10, y2, r2, [1.0]),
            (10, y2, r3, [0.0]),
            (3, y7, y7, [float("nan")]),
            (10, y3, r3, [float("nan")]),
            (10, y3, r2, [float("nan")]),
            (9, y2, r1, [0.5]),
            (1, y4, r4, [1.0 / 6]),
            (1, y1, [r1, r2], [0.5, 0.0]),
            (1, [y1, y2], [r1, r2], [0.5, 0.5]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = Recall(k).score(y_gold, y_pred, nan_handling="propagate").tolist()

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
        pytrec_result = RelevanceEvaluator(qrel, ["recall.5"]).evaluate(run)

        scores = Rankings.from_scores(scores)
        labels = SparseBinaryLabels.from_matrix(obs)
        results = create_results({"recall_5": Recall(5).score(labels, scores)})

        for query_id, metrics in results.items():
            for metric_name, value in metrics.items():
                assert value == pytest.approx(
                    pytrec_result[query_id][metric_name], rel=1e-3
                )
