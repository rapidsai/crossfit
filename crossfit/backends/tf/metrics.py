# import abc
from dataclasses import fields, asdict

# import functools
from typing import TypeVar, Generic

# from typing import Optional, TypeVar, Generic, Type, overload

import tensorflow as tf

from crossfit.calculate.aggregate import Aggregator
from crossfit.data import crossarray
from crossfit.metrics.base import CrossMetric

# from crossfit.core.metric import ComparisonMetric, StateType, Array
# from crossfit.stats.continuous.common import AverageState

TFMetricType = TypeVar("TFMetricType", bound=tf.keras.metrics.Metric)
AggregatorType = TypeVar("AggregatorType", bound=Aggregator)


class AggregatorMetric(tf.keras.metrics.Metric, Generic[AggregatorType]):
    def __init__(self, aggregator: AggregatorType, name=None, **kwargs):
        name = name or aggregator.__class__.__name__
        super().__init__(name=name, **kwargs)
        self.aggregator = aggregator

    def build(self, input_shape):
        for field in fields(self.aggregator.state):
            setattr(self, field.name, self.add_weight(field.name, initializer="zeros"))
        return super().build(input_shape)

    def prepare(self, y_true, y_pred, sample_weight=None):
        with crossarray:
            self.build(y_true.shape)
            batch = self.aggregator(y_true, y_pred, sample_weight=sample_weight)
            current_state = {
                field.name: tf.convert_to_tensor(getattr(self, field.name))
                for field in fields(self.aggregator.state)
            }

            updated_state = batch.combine(self.aggregator.state(**current_state))
            for key, val in asdict(updated_state).items():
                variable = getattr(self, key)
                variable.assign(tf.cast(val, variable.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.prepare(y_true, y_pred, sample_weight)

    def result(self):
        state_dict = {}

        for field in fields(self.aggregator.state):
            state_dict[field.name] = getattr(self, field.name)

        state = self.aggregator.state(**state_dict)
        outputs = self.aggregator.present(state)

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs


def to_tf_metric(aggregator: AggregatorType) -> AggregatorMetric[AggregatorType]:
    if isinstance(aggregator, CrossMetric):
        aggregator = aggregator.to_aggregator()
    return AggregatorMetric(aggregator)


# class TFMetric(Generic[TFMetricType, StateType], ComparisonMetric[StateType], abc.ABC):
#     @abc.abstractmethod
#     def metric(self) -> TFMetricType:
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def present_metric(
#         self,
#         metric: TFMetricType,
#         array_type: Type[Array],
#     ) -> StateType:
#         raise NotImplementedError()

#     def prepare(
#         self,
#         data: Array,
#         comparison: Array,
#         sample_weight: Array = None,
#     ) -> StateType:
#         metric = self.metric()
#         metric.update_state(
#             convert_array(data, tf.Tensor),
#             convert_array(comparison, tf.Tensor),
#             sample_weight=self.parse_sample_weight(sample_weight),
#         )

#         return self.present_metric(metric, array_type=type(data))

#     def parse_sample_weight(
#         self, sample_weight: Optional[Array]
#     ) -> Optional[tf.Tensor]:
#         if sample_weight is None:
#             return None

#         return convert_array(sample_weight, tf.Tensor)


# class TFMeanMetric(
#     Generic[TFMetricType], TFMetric[TFMetricType, AverageState], abc.ABC
# ):
#     def prepare(
#         self,
#         data: Array,
#         comparison: Array,
#         sample_weight: Array = None,
#     ) -> AverageState:
#         return super().prepare(data, comparison, sample_weight=sample_weight)

#     def present_metric(
#         self,
#         metric: tf.keras.metrics.Metric,
#         array_type: Type[Array],
#     ) -> AverageState:
#         return AverageState(
#             count=convert_array(tf.convert_to_tensor(metric.count), array_type),
#             sum=convert_array(tf.convert_to_tensor(metric.total), array_type),
#         )


# @overload
# def mean_metric(metric: Type[tf.keras.metrics.Metric]):
#     ...


# @overload
# def mean_metric(metric: tf.keras.metrics.Metric):
#     ...


# def mean_metric(metric):
#     if isinstance(metric, type):
#         return _mean_metric_decorator(metric)

#     class DynamicMetric(TFMeanMetric):
#         def metric(self) -> tf.keras.metrics.Metric:
#             return metric

#     DynamicMetric.__name__ = metric.__class__.__name__

#     return DynamicMetric()


# def _mean_metric_decorator(metric: Type[TFMetricType]):
#     @functools.wraps(metric)
#     def metric_wrapper(*args, **kwargs):
#         tf_metric = metric(*args, **kwargs)

#         return mean_metric(tf_metric)

#     return metric_wrapper


# TFMeanRelativeError = mean_metric(tf.keras.metrics.MeanRelativeError)
# TFAccuracy = mean_metric(tf.keras.metrics.Accuracy)
# TFBinaryAccuracy = mean_metric(tf.keras.metrics.BinaryAccuracy)
# TFCategoricalAccuracy = mean_metric(tf.keras.metrics.CategoricalAccuracy)
# TFSparseCategoricalAccuracy = mean_metric(tf.keras.metrics.SparseCategoricalAccuracy)
# TFTopKCategoricalAccuracy = mean_metric(tf.keras.metrics.TopKCategoricalAccuracy)
# TFSparseTopKCategoricalAccuracy = mean_metric(
#     tf.keras.metrics.SparseTopKCategoricalAccuracy
# )
# TFCosineSimilarity = mean_metric(tf.keras.metrics.CosineSimilarity)
# TFMeanAbsoluteError = mean_metric(tf.keras.metrics.MeanAbsoluteError)
# TFMeanAbsolutePercentageError = mean_metric(
#     tf.keras.metrics.MeanAbsolutePercentageError
# )
# TFMeanSquaredError = mean_metric(tf.keras.metrics.MeanSquaredError)
# TFMeanSquaredLogarithmicError = mean_metric(
#     tf.keras.metrics.MeanSquaredLogarithmicError
# )
# TFHinge = mean_metric(tf.keras.metrics.Hinge)
# TFSquaredHinge = mean_metric(tf.keras.metrics.SquaredHinge)
# TFCategoricalHinge = mean_metric(tf.keras.metrics.CategoricalHinge)
# TFRootMeanSquaredError = mean_metric(tf.keras.metrics.RootMeanSquaredError)
# TFLogCoshError = mean_metric(tf.keras.metrics.LogCoshError)
# TFPoisson = mean_metric(tf.keras.metrics.Poisson)
# TFKLDivergence = mean_metric(tf.keras.metrics.KLDivergence)
# TFCategoricalCrossentropy = mean_metric(tf.keras.metrics.CategoricalCrossentropy)
# TFSparseCategoricalCrossentropy = mean_metric(
#     tf.keras.metrics.SparseCategoricalCrossentropy
# )
