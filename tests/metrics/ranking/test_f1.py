import pytest

pytest.importorskip("cupy")

from crossfit.data.sparse.ranking import SparseBinaryLabels, SparseRankings
from crossfit.metric.ranking import F1


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


class TestF1:
    @pytest.mark.parametrize(
        "k,y_gold,y_pred,expect",
        [
            (3, y2, r1, [0.0]),
            (10, y2, r1, [1.0 / 3]),
            (2, y2, r2, [1.0]),
            (10, y2, r3, [0.0]),
            (1, y4, r4, [0.285714286]),
            (10, y3, r3, [float("nan")]),
            (10, y3, r2, [float("nan")]),
            (1, y1, [r1, r2], [2.0 / 3, 0.0]),
            (1, [y1, y1], [r1, r2], [2.0 / 3, 0.0]),
        ],
    )
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = SparseBinaryLabels.from_positive_indices(y_gold)
        if len(y_pred) == 0 or [] in y_pred:
            with pytest.warns(UserWarning):
                y_pred = SparseRankings.from_ranked_indices(y_pred)
        else:
            y_pred = SparseRankings.from_ranked_indices(y_pred)
        pred = F1(k).score(y_gold, y_pred, nan_handling="propagate").tolist()
        assert pred == pytest.approx(expect, nan_ok=True)
