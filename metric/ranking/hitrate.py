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

    def score(self, y_true, y_pred):
        r"""
        Computes HitRate@k [MN]_ as the whether a relevant item occurs
        in *k* results of `y_pred`.
        Differs from Recall@k in that `y_true` has to contain exactly
        one element per row.
        More formally, the HitRate of each ranking *y* in `y_pred` with labels
        `y_true` is defined as

        .. math::

                \mathrm{HitRate}@k(y) &= \sum_{i=1}^{k'} \mathrm{rel}(y_i),

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary and exactly one relevant item per row.
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

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`.
        Per definition above, :math:`\mathrm{HitRate}@k(y) = 0`.

        2. There is not exactly one relevant item in
        `y_true`: :math:`\mathrm{HitRate}@k(y) =` NaN.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, HitRate
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> HitRate(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([0., 1.])

        """
        return super().score(y_true, y_pred)

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        scores = self._recall(y_true, y_pred_labels)
        scores[n_pos != 1] = np.NaN  # Not defined for no or multiple positives
        return scores
