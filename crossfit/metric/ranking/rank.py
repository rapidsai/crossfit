import numpy as np

from crossfit.data.array.masked import MaskedArray
from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels


class ReciprocalRank(BinaryRankingMetric):
    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        # It is 1/rank if document appears in top k, 0 otherwise
        scores = np.max(labels / ranks, axis=-1, initial=0.0)
        scores[n_pos == 0] = np.NaN  # Not defined for no multiple positives

        return scores


class MeanRanks(BinaryRankingMetric):
    def __init__(self):
        self._k = None

    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels.filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        scores = np.sum(ranks * labels, axis=-1)
        scores[n_pos > 0] = scores[n_pos > 0] / n_pos[n_pos > 0]
        scores[n_pos == 0] = np.NaN
        return scores


class FirstRelevantRank(BinaryRankingMetric):
    def __init__(self):
        self._k = None

    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels.filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        ranks = ranks * labels
        ranks[ranks == 0] = np.inf
        scores = np.min(ranks, axis=-1)
        scores[n_pos == 0] = np.NaN

        return scores
