import numpy as np

from crossfit.metric.ranking.recall import Recall


class HitRate(Recall):
    """
    Parameters
    ----------
    k : int
            specifies number of top results `k` of each ranking to be evaluated.

    Raises
    ------
    ValueError
            if `k` is not integer > 0.
    """

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        scores = self._recall(y_true, y_pred_labels)
        scores[n_pos != 1] = np.NaN  # Not defined for no or multiple positives
        return scores
