import numpy as np

from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels
from crossfit.data.array.masked import MaskedArray


class Recall(BinaryRankingMetric):
    def __init__(self, k, truncated=False):
        super().__init__(k)
        self._truncated = truncated

    def _recall(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(
            (y_pred_labels.data[:, : self._k] == 1) & (~y_pred_labels.mask[:, : self._k]), axis=-1
        )

        scores = np.NaN * np.zeros_like(n_relevant, dtype=float)
        if self._truncated:
            denominator = np.clip(n_pos[n_pos > 0], None, self._k)
        else:
            denominator = n_pos[n_pos > 0]
        scores[n_pos > 0] = n_relevant[n_pos > 0].astype(float) / denominator

        return scores

    def _score(self, y_true, y_pred_labels):
        return self._recall(y_true, y_pred_labels)
