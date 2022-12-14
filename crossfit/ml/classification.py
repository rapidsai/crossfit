from dataclasses import dataclass

import crossfit.array as cnp
import numpy as np

from crossfit.core.metric import Array, ComparisonMetric, MetricState, field


@dataclass
class BinaryClfState(MetricState):
    tp: Array = field(combine=sum)
    tn: Array = field(combine=sum)
    fp: Array = field(combine=sum)
    fn: Array = field(combine=sum)

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.count

    @property
    def precision(self):
        # calculate precision
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        # calculate recall
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self):
        # calculate f1-score
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall)

    @property
    def tpr(self):
        return self.tp / (self.tp + self.fn)

    @property
    def fpr(self):
        return self.fp / (self.fp + self.tn)

    @property
    def count(self):
        return self.tp + self.fp + self.tn + self.fn

    @property
    def auc(self):
        return np.trapz([self.tpr, self.fpr], [0, 1])

    def plot_confusion_matrix(self, colorscale="blues"):
        import plotly.figure_factory as ff

        figure = ff.create_annotated_heatmap(
            [[self.tp, self.fp], [self.fn, self.tn]],
            x=["Predicted Positive", "Predicted Negative"],
            y=["Actual Positive", "Actual Negative"],
            colorscale=colorscale,
        )
        figure.show()


class BinaryMetrics(ComparisonMetric[BinaryClfState]):
    def __init__(self, threshold: float = 0.5, apply_threshold=True):
        self.threshold = threshold
        self.apply_threshold = apply_threshold

    def prepare(self, data: Array, comparison: Array, **kwargs) -> BinaryClfState:
        if self.apply_threshold:
            data = data > self.threshold

        tp, tn, fp, fn = calculate_binary_classification_metrics(data, comparison)
        return BinaryClfState(tp=tp, fp=fp, tn=tn, fn=fn)


@dataclass
class BinaryCurveState(MetricState):
    thresholds: Array = field(is_list=True)
    tp: Array = field(is_list=True)
    fp: Array = field(is_list=True)

    def auc(self, curve="roc"):
        from sklearn import metrics

        if curve == "roc":
            x, y, _ = self.roc_curve()
        elif curve == "pr":
            x, y, _ = self.pr_curve()
        else:
            raise ValueError(
                f"Unknown curve: {curve}. Possible values are 'roc' or 'pr'"
            )

        return metrics.auc(x, y)

    def roc_curve(self, drop_intermediate=True):
        # Attempt to drop thresholds corresponding to points in between and
        # collinear with other points. These are always suboptimal and do not
        # appear on a plotted ROC curve (and thus do not affect the AUC).
        # Here np.diff(_, 2) is used as a "second derivative" to tell if there
        # is a corner at the point. Both fps and tps must be tested to handle
        # thresholds with multiple data points (which are combined in
        # _binary_clf_curve). This keeps all cases where the point should be kept,
        # but does not drop more complicated cases like fps = [1, 3, 7],
        # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
        if drop_intermediate and len(self.fp) > 2:
            optimal_idxs = np.where(
                np.r_[
                    True, np.logical_or(np.diff(self.fp, 2), np.diff(self.tp, 2)), True
                ]
            )[0]
            fps = self.fp[optimal_idxs]
            tps = self.tp[optimal_idxs]
            thresholds = self.thresholds[optimal_idxs]

        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

        if fps[-1] <= 0:
            # warnings.warn(
            #     "No negative samples in y_true, false positive value should be meaningless",
            #     UndefinedMetricWarning,
            # )
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            # warnings.warn(
            #     "No positive samples in y_true, true positive value should be meaningless",
            #     UndefinedMetricWarning,
            # )
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        return fpr, tpr, thresholds

    def plot_roc_curve(self, plotly=True, **kwargs):
        x, y, _ = self.roc_curve()
        if plotly:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="ROC"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
            fig.update_layout(**kwargs)
            fig.show()

        from sklearn import metrics

        display = metrics.RocCurveDisplay(fpr=x, tpr=y, roc_auc=metrics.auc(x, y))

        return display.plot(**kwargs)

    def pr_curve(self):
        ps = self.tp + self.fp
        # Initialize the result array with zeros to make sure that precision[ps == 0]
        # does not contain uninitialized values.
        precision = np.zeros_like(self.tp)
        np.divide(self.tp, ps, out=precision, where=(ps != 0))

        # When no positive label in y_true, recall is set to 1 for all thresholds
        # tps[-1] == 0 <=> y_true == all negative labels
        if self.tp[-1] == 0:
            # warnings.warn(
            #     "No positive class found in y_true, "
            #     "recall is set to one for all thresholds."
            # )
            recall = np.ones_like(self.tp)
        else:
            recall = self.tp / self.tp[-1]

        # reverse the outputs so recall is decreasing
        sl = slice(None, None, -1)
        return (
            np.hstack((precision[sl], 1)),
            np.hstack((recall[sl], 0)),
            self.thresholds[sl],
        )

    def plot_pr_curve(self, plotly=True, **kwargs):
        x, y, _ = self.pr_curve()
        if plotly:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PR"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
            fig.update_layout(**kwargs)
            fig.show()

        from sklearn import metrics

        display = metrics.PrecisionRecallDisplay(precision=x, recall=y)

        display.plot()


def calculate_binary_classification_metrics(predictions, labels):
    # Calculate true-positives
    tp = cnp.sum((predictions == 1) & (labels == 1))

    # Calculate true-negatives
    tn = cnp.sum((predictions == 0) & (labels == 0))

    # Calculate false-positives
    fp = cnp.sum((predictions == 1) & (labels == 0))

    # Calculate false-negatives
    fn = cnp.sum((predictions == 0) & (labels == 1))

    return tp, tn, fp, fn
