import abc
import functools
from typing import Optional, TypeVar, Generic, Type, overload

import tensorflow as tf

from crossfit.array import convert
from crossfit.core.metric import ComparisonMetric, StateType, Array
from crossfit.stats.continuous.common import AverageState

TFMetricType = TypeVar("TFMetricType", bound=tf.keras.metrics.Metric)


class TFMetric(Generic[TFMetricType, StateType], ComparisonMetric[StateType], abc.ABC):
    @abc.abstractmethod
    def metric(self) -> TFMetricType:
        raise NotImplementedError()

    @abc.abstractmethod
    def present_metric(
        self,
        metric: TFMetricType,
        array_type: Type[Array],
    ) -> StateType:
        raise NotImplementedError()

    def prepare(
        self,
        data: Array,
        comparison: Array,
        sample_weight: Optional[Array] = None,
    ) -> StateType:
        metric = self.metric()
        metric.update_state(
            convert(data, tf.Tensor),
            convert(comparison, tf.Tensor),
            sample_weight=self.parse_sample_weight(sample_weight),
        )

        return self.present_metric(metric, array_type=type(data))

    def parse_sample_weight(
        self, sample_weight: Optional[Array]
    ) -> Optional[tf.Tensor]:
        if sample_weight is None:
            return None

        return convert(sample_weight, tf.Tensor)


class TFMeanMetric(
    Generic[TFMetricType], TFMetric[TFMetricType, AverageState], abc.ABC
):
    def prepare(
        self,
        data: Array,
        comparison: Array,
        sample_weight: Array = None,
    ) -> AverageState:
        return super().prepare(data, comparison, sample_weight=sample_weight)

    def present_metric(
        self,
        metric: tf.keras.metrics.Metric,
        array_type: Type[Array],
    ) -> AverageState:
        return AverageState(
            count=convert(tf.convert_to_tensor(metric.count), array_type),
            sum=convert(tf.convert_to_tensor(metric.total), array_type),
        )


@overload
def mean_metric(metric: Type[tf.keras.metrics.Metric]):
    ...


@overload
def mean_metric(metric: tf.keras.metrics.Metric):
    ...


def mean_metric(metric):
    if isinstance(metric, type):
        return _mean_metric_decorator(metric)

    class DynamicMetric(TFMeanMetric):
        def metric(self) -> tf.keras.metrics.Metric:
            return metric

    DynamicMetric.__name__ = metric.__class__.__name__

    return DynamicMetric()


def _mean_metric_decorator(metric: Type[TFMetricType]):
    @functools.wraps(metric)
    def metric_wrapper(*args, **kwargs):
        tf_metric = metric(*args, **kwargs)

        return mean_metric(tf_metric)

    return metric_wrapper


TFMeanRelativeError = mean_metric(tf.keras.metrics.MeanRelativeError)
TFAccuracy = mean_metric(tf.keras.metrics.Accuracy)
TFBinaryAccuracy = mean_metric(tf.keras.metrics.BinaryAccuracy)
TFCategoricalAccuracy = mean_metric(tf.keras.metrics.CategoricalAccuracy)
TFSparseCategoricalAccuracy = mean_metric(tf.keras.metrics.SparseCategoricalAccuracy)
TFTopKCategoricalAccuracy = mean_metric(tf.keras.metrics.TopKCategoricalAccuracy)
TFSparseTopKCategoricalAccuracy = mean_metric(
    tf.keras.metrics.SparseTopKCategoricalAccuracy
)
TFCosineSimilarity = mean_metric(tf.keras.metrics.CosineSimilarity)
TFMeanAbsoluteError = mean_metric(tf.keras.metrics.MeanAbsoluteError)
TFMeanAbsolutePercentageError = mean_metric(
    tf.keras.metrics.MeanAbsolutePercentageError
)
TFMeanSquaredError = mean_metric(tf.keras.metrics.MeanSquaredError)
TFMeanSquaredLogarithmicError = mean_metric(
    tf.keras.metrics.MeanSquaredLogarithmicError
)
TFHinge = mean_metric(tf.keras.metrics.Hinge)
TFSquaredHinge = mean_metric(tf.keras.metrics.SquaredHinge)
TFCategoricalHinge = mean_metric(tf.keras.metrics.CategoricalHinge)
TFRootMeanSquaredError = mean_metric(tf.keras.metrics.RootMeanSquaredError)
TFLogCoshError = mean_metric(tf.keras.metrics.LogCoshError)
TFPoisson = mean_metric(tf.keras.metrics.Poisson)
TFKLDivergence = mean_metric(tf.keras.metrics.KLDivergence)
TFCategoricalCrossentropy = mean_metric(tf.keras.metrics.CategoricalCrossentropy)
TFSparseCategoricalCrossentropy = mean_metric(
    tf.keras.metrics.SparseCategoricalCrossentropy
)


class TFAUC(TFMetric):
    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
    ):
        self.num_thresholds = num_thresholds
        self.curve = curve
        self.summation_method = summation_method
        self.thresholds = thresholds
        self.multi_label = multi_label
        self.num_labels = num_labels
        self.label_weights = label_weights
        self.from_logits = from_logits

    def metric(self) -> tf.keras.metrics.AUC:
        return tf.keras.metrics.AUC(
            num_thresholds=self.num_thresholds,
            curve=self.curve,
            summation_method=self.summation_method,
            thresholds=self.thresholds,
            multi_label=self.multi_label,
            num_labels=self.num_labels,
            label_weights=self.label_weights,
            from_logits=self.from_logits,
        )

    def present_metric(
        self,
        metric: TFMetricType,
        array_type: Type[Array],
    ) -> StateType:
        raise NotImplementedError()
