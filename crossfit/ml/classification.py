from dataclasses import dataclass

import crossfit.array as cnp
import numpy as np

from crossfit.core.metric import Array, ComparisonMetric, MetricState, field


@dataclass
class BinaryClassificationState(MetricState):
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


class BinaryMetrics(ComparisonMetric[BinaryClassificationState]):
    def __init__(self, threshold: float = 0.5, apply_threshold=True):
        self.threshold = threshold
        self.apply_threshold = apply_threshold

    def prepare(
        self, data: Array, comparison: Array, **kwargs
    ) -> BinaryClassificationState:
        if self.apply_threshold:
            data = data > self.threshold

        tp, tn, fp, fn = calculate_binary_classification_metrics(data, comparison)
        return BinaryClassificationState(tp=tp, fp=fp, tn=tn, fn=fn)


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
