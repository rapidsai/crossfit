import numpy as np

from crossfit.data.array.masked import MaskedArray
from crossfit.metric.continuous.mean import Mean
from crossfit.data.sparse.ranking import BinaryLabels, Labels, Rankings


class RankingMetric(Mean):
    def prepare(
        self,
    ):
        ...

    def score(self, y_true: Labels, y_pred: Rankings):
        """
        Individual scores for each ranking.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.
        ValueError
                if `n_bootstrap_samples`, `confidence` or `nan_handling` contain invalid values.
        """
        if not isinstance(y_true, Labels):
            raise TypeError("y_true must be of type Labels")
        if not isinstance(y_pred, Rankings):
            raise TypeError("y_pred must be of type Rankings")

        y_pred_labels = y_true.get_labels_for(y_pred, self._k)

        return self._score(y_true, y_pred_labels)

    @classmethod
    def _bootstrap_ci(cls, scores, n_bootstrap_samples, confidence):
        if not isinstance(n_bootstrap_samples, int) or n_bootstrap_samples <= 1:
            raise ValueError("n_bootstrap_samples must be int > 1")
        elif not isinstance(confidence, float) or confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("Confidence must be float and 0 < confidence < 1")

        if len(scores):
            resamples = np.random.choice(scores, (len(scores), n_bootstrap_samples), replace=True)
            bootstrap_means = resamples.mean(axis=0)

            # Compute "percentile bootstrap"
            alpha_2 = (1 - confidence) / 2.0
            lower_ci = np.quantile(bootstrap_means, alpha_2)
            upper_ci = np.quantile(bootstrap_means, 1.0 - alpha_2)
            return (lower_ci, upper_ci)
        else:
            return (float("nan"), float("nan"))

    def mean(
        self,
        y_true,
        y_pred,
        nan_handling="propagate",
        conf_interval=False,
        n_bootstrap_samples=1000,
        confidence=0.95,
    ):
        r"""
        Mean score over all ranking after handling NaN values.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, see also above.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated.
        nan_handling : {'propagate', 'drop', 'zerofill'}, optional
                `'propagate'` (default):
                        Return NaN if any value is NaN
                `'drop'` :
                        Ignore NaN values
                `'zerofill'` :
                        Replace NaN values with zero
        conv_interval : bool, optional
                If True, then return bootstrapped confidence intervals of mean,
                otherwise interval is None.
                Defaults to False.
        n_bootstrap_samples : int, optional
                Number of bootstrap samples to draw.
        confidence : float, optional
                Indicates width of confidence interval. Default is 0.95 (95%).
        Returns
        -------
        mean: float `mean` if `conv_interval` is `false` otherwise
                Dictionary with ``mean["score"]`` and ``mean["conf_interval"]``
                for the confidence interval tuple `(lower CI, upper CI)`.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.
        ValueError
                if `n_bootstrap_samples`, `confidence` or `nan_handling` contain invalid
                values.
        """

        scores = self.score(y_true, y_pred)
        if nan_handling == "drop":
            scores = scores[~np.isnan(scores)]
        elif nan_handling == "zerofill":
            scores = np.nan_to_num(scores)
        elif nan_handling == "propagate":
            if np.isnan(scores).sum():
                scores = []
        else:
            raise ValueError('nan_handling must be "propagate", "drop" or "zerofill"')

        if conf_interval:
            ci = self._bootstrap_ci(scores, n_bootstrap_samples, confidence)
        else:
            ci = None

        if len(scores):
            mean = scores.mean()
        else:
            mean = float("nan")

        if conf_interval:
            return {"score": mean, "conf_interval": ci}
        else:
            return mean

    def name(self):
        if self._k is None:
            k = ""
        else:
            k = f"@{self._k}"

        return self.__class__.__name__ + k


class BinaryRankingMetric(RankingMetric):
    def __init__(self, k):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Cutoff k needs to be integer > 0")
        self._k = k

    def score(self, y_true: BinaryLabels, y_pred: MaskedArray):
        if not isinstance(y_true, BinaryLabels):
            raise TypeError(
                f"y_true must be of type BinaryLabels but is of instance {type(y_true)}"
            )
        return super().score(y_true, y_pred)
