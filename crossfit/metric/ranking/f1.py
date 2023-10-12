import numpy as np

from crossfit.data.array.masked import MaskedArray
from crossfit.metric.ranking.base import SparseBinaryLabels
from crossfit.metric.ranking.precision import Precision
from crossfit.metric.ranking.recall import Recall


class F1(Precision, Recall):
    def _score(self, y_true, y_pred_labels):
        recall = self._recall(y_true, y_pred_labels)
        precision = self._precision(y_true, y_pred_labels)

        product = 2 * recall * precision
        sm = recall + precision

        # return 0 for geometric mean if both are zero
        scores = np.zeros_like(product, dtype=float)
        valid = np.nan_to_num(product) > 0
        invalid = np.isnan(product)

        scores[valid] = product[valid] / sm[valid]
        scores[invalid] = np.NaN

        return scores
