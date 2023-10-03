import numpy as np

from crossfit.metric.ranking.base import RankingMetric, SparseLabels, SparseRankings
from crossfit.data.array.masked import MaskedArray
from crossfit.data.array.conversion import convert_array


class DCG(RankingMetric):
    """


    Parameters
    ----------
    k : int
            specifies number of top results `k` of each ranking to be evaluated.

    relevance_scaling : str, ['binary' (default), 'power']
            Determines are relevance labels are transformed:

            `'identity'`: (default)
                    :math:`f(\mathrm{rel}(y_i)) = \mathrm{rel}(y_i)`
            `'power'`:
                    :math:`f(\mathrm{rel}(y_i)) = 2^{\mathrm{rel}(y_i)} - 1`

    log_base : str, ['e' (default), '2']
            Determines what log base is used in denominator.
            The smaller this value, the heavier emphasis on top-ranked documents.

            `'e'` (default):
                    Natural logarithm :math:`\ln`
            `'2'`:
                    :math:`\log_2`

    Notes
    -----
    The original definition of (n)DCG [KJ]_ uses 'identity' for `relevance_scaling`,
    but leaves the choice of `log_base` open.

    Raises
    ------
    ValueError
            if `k` is not integer > 0 or `relevance_scaling` or `log_base` are invalid.

    """

    SCALERS = {"identity": lambda x: x, "power": lambda x: np.power(x, 2) - 1}
    LOGS = {"2": lambda x: np.log2(x), "e": lambda x: np.log(x)}

    def __init__(self, k=None, relevance_scaling="identity", log_base="e"):
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

    def score(self, y_true: SparseLabels, y_pred: SparseRankings):
        r"""
        Computes Discounted Cumulative Gain at k (DCG@k) [KJ]_ as a
        weighted sum of relevance labels of top *k* results of `y_pred`.
        More formally, the recall of each ranking *y* in `y_pred`
        with labels `y_true` is defined as

        .. math::

                \mathrm{DCG}@k(y) &= \sum_{i=1}^{k'} \frac{f(\mathrm{rel}(y_i))}{\log_b(i + 1)},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of
        the item at rank *i* in the ranking *y*.
        *f* is the `relevance_scaling` function and *b* the `log_base`
        parameters defined earlier.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, binary or numeric.
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
        Per definition above, :math:`\mathrm{DCG}@k(y) = 0`.

        2. There are no items with relevance > 0 in `y_true`:
        :math:`\mathrm{DCG}@k(y) =` NaN to make it consistent with other metrics.

        Examples
        --------
        >>> from rankereval import NumericLabels, Rankings, DCG
        >>> # use separate labels for each ranking
        >>> y_true = NumericLabels.from_matrix([[1, 2, 3], [4, 5]])
        >>> y_pred = Rankings.from_ranked_indices([[0,2,1], [1,0]])
        >>> DCG(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([ 5.61610776, 10.85443211])

        """
        return super().score(y_true, y_pred)


class NDCG(DCG):
    """
    For a description of the arguments, see :class:`DCG`.
    """

    def _score(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        dcg = self._dcg(y_true, y_pred_labels)
        ideal_labels = y_true.get_labels_for(y_true.as_rankings(), self._k)
        idcg = self._dcg(y_true, ideal_labels)

        return dcg / idcg

    def score(self, y_true: SparseLabels, y_pred: SparseRankings):
        r"""
        Computes the *normalized* Discounted Cumulative Gain at k (nDCG@k) [KJ]_
        as a weighted sum of relevance labels of top *k* results of `y_pred`,
        normalized to the range [0, 1].
        More formally, the nDCG of each ranking *y* in `y_pred` with
        labels `y_true` is defined as

        .. math::

                \mathrm{nDCG}@k(y)  = \begin{cases} \frac{\mathrm{DCG}@k(y)}
                {\mathrm{IDCG}@k(y)} &\mbox{if } \mathrm{IDCG}@k(y) > 0 \\
                0 & \mbox{otherwise } \end{cases},

        where :math:`\mathrm{IDCG}@k(y)` is the maximum DCG@k value that can
        be achieved on *all* relevance labels (i.e., DCG@k of the sorted relevance labels).

        .. note::

                Sometimes IDCG is defined with respect to only
                the retrieved or recommended set of items *y*. This is not desirable
                as nDCG could be artificially inflated, e.g., by
                returning only one relevant item at the top
                and then filling up the ranking with and irrelevant items.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, binary or numeric.
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

        1. `y` is empty, i.e., :math:`|y|=0`. Per definition above, :math:`\mathrm{DCG}@k(y) = 0`.

        2. There are no items with relevance > 0 in `y_true`: :math:`\mathrm{DCG}@k(y) =` NaN
        to make it consistent with other metrics.

        3. There are no items with relevance > 0 in `y` up to *k*:  :math:`\mathrm{nDCG}@k(y) = 0`.

        Examples
        --------
        >>> from rankereval import NumericLabels, Rankings, NDCG
        >>> # use separate labels for each ranking
        >>> y_true = NumericLabels.from_matrix([[1, 2, 3], [4, 5]])
        >>> y_pred = Rankings.from_ranked_indices([[0,2,1], [1,0]])
        >>> NDCG(3).score(y_true, y_pred) # doctest: +NORMALIZE_WHITESPACE
        array([0.81749351, 1.        ])

        """
        return super().score(y_true, y_pred)
