import numpy as np

from crossfit.data.array.masked import MaskedArray
from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels


class Precision(BinaryRankingMetric):
    def __init__(self, k, truncated=False):
        super().__init__(k)
        self._truncated = truncated

    def _precision(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(
            (y_pred_labels.data[:, : self._k] == 1)
            & (~y_pred_labels.mask[:, : self._k]),
            axis=-1,
        )

        if self._truncated:
            items = np.broadcast_to(~y_pred_labels.mask, y_pred_labels.shape)
            n_items_in_y_pred = items.sum(axis=1).flatten()

            # not defined if there are no relevant labels
            scores = np.NaN * np.zeros_like(n_relevant, dtype=float)
            valid = (n_items_in_y_pred > 0) & (n_pos > 0)

            scores[valid] = n_relevant[valid].astype(float) / np.minimum(
                n_items_in_y_pred[valid], self._k
            )
        else:
            scores = n_relevant.astype(float) / self._k
            # not defined if there are no relevant labels
            scores[n_pos == 0] = np.NaN

        return scores

    def _score(self, y_true, y_pred_labels):
        return self._precision(y_true, y_pred_labels)


class AP(BinaryRankingMetric):
    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)

        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        precision = np.cumsum(labels, axis=-1) / ranks

        scores = np.zeros_like(n_pos, dtype=float)
        scores[n_pos > 0] = np.sum(precision * labels, axis=-1)[n_pos > 0] / np.clip(
            n_pos[n_pos > 0], None, self._k
        )
        scores[n_pos == 0] = np.NaN

        return scores
