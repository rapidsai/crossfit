import numpy as np

from crossfit.metric.ranking.base import RankingMetric, SparseLabels, SparseRankings
from crossfit.data.array.masked import MaskedArray
from crossfit.data.array.conversion import convert_array


class DCG(RankingMetric):
    SCALERS = {"identity": lambda x: x, "power": lambda x: np.power(x, 2) - 1}
    LOGS = {"2": lambda x: np.log2(x), "e": lambda x: np.log(x)}

    def __init__(self, k=None, relevance_scaling="identity", log_base="2"):
        self._k = k
        if relevance_scaling not in self.SCALERS:
            raise ValueError("Relevance scaling must be 'identity' or 'power'.")
        if log_base not in self.LOGS:
            raise ValueError("Log base needs to be 'e' or '2'.")
        self._rel_scale = self.SCALERS[relevance_scaling]
        self._log_fct = self.LOGS[log_base]

    def _dcg(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)
        ranks = convert_array(ranks, type(y_pred_labels.data))

        scores = np.sum(self._rel_scale(labels) / self._log_fct(ranks + 1), axis=-1)
        scores[n_pos == 0] = np.NaN
        return scores

    def _score(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        return self._dcg(y_true, y_pred_labels)


class NDCG(DCG):
    def _score(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        dcg = self._dcg(y_true, y_pred_labels)
        ideal_labels = y_true.get_labels_for(y_true.as_rankings(), self._k)
        idcg = self._dcg(y_true, ideal_labels)

        return dcg / idcg
