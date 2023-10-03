import numpy as np

from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels
from crossfit.data.array.masked import MaskedArray


class Recall(BinaryRankingMetric):
    """
    Parameters
    ----------
    k : int
            specifies number of top results `k` of each ranking to be evaluated.
    truncated : bool
            if `true`, number of relevant results gets clipped at `k`.
    Raises
    ------
    ValueError
            if `k` is not integer > 0.
    """

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

    def score(self, y_true, y_pred):
        r"""
        Computes Recall@k [MN]_ as the fraction of relevant results in `y_true` that were
        in the top *k* results of `y_pred`.
        More formally, the recall of each ranking *y* in `y_pred` with
        labels `y_true` is defined as

        .. math::

                \mathrm{Recall}@k(y) =
                \frac{\sum_{i=1}^{k'} \mathrm{rel}(y_i)}{n_{rel}},

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y* and :math:`n_{rel} = \left\| y_{true} \right\| _1` if `truncated`
        is false (default) and :math:`n_{rel} = \max(\left\| y_{true} \right\| _1, k)` otherwise.

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

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. Per definition above,
        :math:`\mathrm{Recall}@k(y) = 0`.

        2. There are no relevant items in `y_true`: :math:`\mathrm{Recall}@k(y) =` NaN. This
        marks invalid instances explicitly.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, Recall
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> Recall(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([0., 1.])

        """
        return super().score(y_true, y_pred)
