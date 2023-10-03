import numpy as np

from crossfit.metric.ranking.base import BinaryRankingMetric, BinaryLabels
from crossfit.data.array.masked import MaskedArray


class Precision(BinaryRankingMetric):
    """
    Parameters
    ----------
    k : int
            specifies number of top results `k` of each ranking to be evaluated.
    truncated : bool
            if `true`, `k` gets clipped at length of input ranking.

    Raises
    ------
    ValueError
            if `k` is not integer > 0.
    """

    def __init__(self, k, truncated=False):
        super().__init__(k)
        self._truncated = truncated

    def _precision(self, y_true: BinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(
            (y_pred_labels.data[:, : self._k] == 1) & (~y_pred_labels.mask[:, : self._k]), axis=-1
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

    def score(self, y_true, y_pred):
        r"""
        Computes Precision@k [MN]_ of each ranking *y* in `y_pred` as

        .. math::

                \mathrm{Precision}@k(y) = \frac{\sum_{i=1}^{k'} \mathrm{rel}(y_i)}{k'},

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*. If `truncated` then :math:`k' = \min(k,|y|)`, i.e., `k'`
        can never exceed the length of the input ranking; otherwise :math:`k' = k` (default).

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
        :math:`\mathrm{Precision}@k(y) = 0`.

        2. There are no relevant items in `y_true`: :math:`\mathrm{Precision}@k(y) =` NaN.
        This marks invalid instances explicitly and is consistent with Recall.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, Precision
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> Precision(3).score(y_true, y_pred)
        array([0.        , 0.66666667])

        """
        return super().score(y_true, y_pred)


class AP(BinaryRankingMetric):
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

    def _score(self, y_true: BinaryLabels, y_pred_labels: MaskedArray):
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

    def score(self, y_true, y_pred):
        r"""
        Computes AveragePrecision@k [MN]_, an approximation to the the area
        under the precision-recall curve, of each ranking *y* in `y_pred` as

        .. math::

                \mathrm{AveragePrecision}@k(y) &= \frac{1}{Z} \sum_{i=1}^{k'}
                \mathrm{rel}(y_i)\cdot \mathrm{Precision}@i(y),

                k' &= \min(k,|y|),

                Z &= \min \big(k, \left\| y_{true} \right\| _1\big);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at
        rank *i* in the ranking *y*.

        .. note::

                Sometimes the denominator *Z* is defined with respect to only the
                retrieved or recommended set of items `y_pred`. This is not desirable
                as AP could be artificially inflated, e.g., by returning only one
                relevant item at the top
                and then filling up the ranking with and irrelevant items.


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

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`.
        Per definition above, :math:`\mathrm{AveragePrecision}@k(y) = 0`.

        2. There are no relevant items in `y_true`:
        :math:`\mathrm{AveragePrecision}@k(y) =` NaN to make it consistent with other metrics.

        3. There are no relevant items in `y_pred` up to *k*:
        Per definition above, :math:`\mathrm{AveragePrecision}@k(y) = 0`.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, AP
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> AP(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([0. , 0.66666667])

        """
        return super().score(y_true, y_pred)
