import numpy as np

from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels
from crossfit.data.array.masked import MaskedArray


class ReciprocalRank(BinaryRankingMetric):
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

    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        # It is 1/rank if document appears in top k, 0 otherwise
        scores = np.max(labels / ranks, axis=-1, initial=0.0)
        scores[n_pos == 0] = np.NaN  # Not defined for no multiple positives

        return scores

    def score(self, y_true, y_pred):
        r"""
        Computes ReciprocalRank@k [NC]_ as the rank where the first relevant item
        occurs in the top *k* results of `y_pred`.
        More formally, the ReciprocalRank of each ranking *y* in `y_pred` is defined as

        .. math::

                \mathrm{ReciprocalRank}@k(y) &= \max_{i=1,\ldots,k'} \frac{\mathrm{rel}(y_i)}{i},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty or first relevant item appears beyond rank *k*.
        Per definition above, :math:`\mathrm{ReciprocalRank}@k(y) = 0`.

        2. There is no relevant item in `y_true`: :math:`\mathrm{ReciprocalRank}@k(y) =` NaN.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, ReciprocalRank
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,0,1], [1,2]])
        >>> ReciprocalRank(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([0.5, 1. ])

        """
        return super().score(y_true, y_pred)


class MeanRanks(BinaryRankingMetric):
    """
    Used for evaluating permutations of `y_true`. Does not accept *k* as it
    scores permutations.
    """

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
    """
    Used for evaluating permutations of `y_true`. Does not accept *k* as it
    scores permutations.
    """

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
