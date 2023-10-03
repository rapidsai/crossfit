import numpy as np

from crossfit.metric.ranking.base import BinaryLabels
from crossfit.metric.ranking.precision import Precision
from crossfit.metric.ranking.recall import Recall
from crossfit.data.array.masked import MaskedArray


class F1(Precision, Recall):
    def score(self, y_true: BinaryLabels, y_pred_labels: MaskedArray):
        r"""
        Computes F1 [MN]_ as harmonic mean of Precision@k and Recall@k.
        More formally, the F1 score of each ranking *y* in `y_pred` is defined as

        .. math::

                \mathrm{F1}@k(y) = \frac{2*\big(\mathrm{Precision}@k(y) *
                \mathrm{Recall}@k(y)\big)}{\mathrm{Precision}@k(y) + \mathrm{Recall}@k(y)}.


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

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. In this case,
        :math:`\mathrm{Recall}@k(y) = \mathrm{Precision}@k(y) = 0` .

        2. If :math:`\mathrm{Recall}@k(y) = \mathrm{Precision}@k(y) = 0`,
        we define :math:`\mathrm{F1}@k(y) = 0`.

        3. There are no relevant items in `y_true`: :math:`\mathrm{F1}@k(y) =` NaN.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, F1
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> F1(3).score(y_true, y_pred)  # doctest: +NORMALIZE_WHITESPACE
        array([0. , 0.5])

        """
        return super().score(y_true, y_pred_labels)

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
