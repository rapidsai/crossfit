import abc
from dataclasses import fields, asdict
from typing import Optional, TypeVar, Generic, Union

import tensorflow as tf

from crossfit.calculate.aggregate import Aggregator
from crossfit.data import crossarray, convert_array
from crossfit.metrics.base import CrossMetric
from crossfit.metrics.mean import Mean

TFMetricType = TypeVar("TFMetricType", bound=tf.keras.metrics.Metric)
AggregatorType = TypeVar("AggregatorType", bound=Aggregator)
CrossMetricType = TypeVar("CrossMetricType", bound=CrossMetric)


class CrossMetricKeras(tf.keras.metrics.Metric, Generic[CrossMetricType]):
    def __init__(
        self, cross_metric: CrossMetricType, name=None, jit_compile=False, **kwargs
    ):
        name = name or cross_metric.__class__.__name__
        super().__init__(name=name, **kwargs)
        self.cross_metric = cross_metric
        self.compute = self.cross_metric
        if jit_compile:
            self.compute = tf.function(self.compute, jit_compile=True)

        # TODO: Should this be in build?
        for field in self.cross_metric.fields():
            setattr(self, field.name, self.add_weight(field.name, initializer="zeros"))

    # def build(self, input_shape):
    #     for field in self.cross_metric.fields():
    #         setattr(self, field.name, self.add_weight(field.name, initializer="zeros"))
    #     return super().build(input_shape)

    def prepare(self, y_true, y_pred, sample_weight=None):
        with crossarray:
            if y_pred.dtype == tf.bool:
                y_pred = tf.cast(y_pred, y_true.dtype)
            batch = self.compute(y_true, y_pred, sample_weight=sample_weight)
            current_state = {
                field.name: tf.convert_to_tensor(getattr(self, field.name))
                for field in self.cross_metric.fields()
            }

            updated_state = batch.combine(self.cross_metric.with_state(**current_state))

            for key, val in updated_state.state_dict.items():
                variable = getattr(self, key)
                variable.assign(tf.cast(val, variable.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.prepare(y_true, y_pred, sample_weight)

    def result(self):
        state_dict = {}

        for field in self.cross_metric.fields():
            state_dict[field.name] = getattr(self, field.name)

        state = self.cross_metric.with_state(**state_dict)
        outputs = state.result

        if isinstance(outputs, dict) and len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs


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


class CrossTFMetricAggregator(
    Aggregator, Generic[TFMetricType, CrossMetricType], abc.ABC
):
    @abc.abstractmethod
    def tf_metric(self) -> TFMetricType:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_cross_metric(
        self,
        metric: TFMetricType,
        array_type,
    ) -> CrossMetricType:
        raise NotImplementedError()

    def prepare(self, y_true, y_pred, sample_weight=None) -> CrossMetricType:
        tf_metric = self.tf_metric()

        tf_true = convert_array(y_true, tf.Tensor)
        tf_pred = convert_array(y_pred, tf.Tensor)
        sample_weight = self.parse_sample_weight(sample_weight)

        if tf_pred.dtype == tf.bool:
            tf_pred = tf.cast(tf_pred, tf_true.dtype)

        tf_metric.update_state(tf_true, tf_pred, sample_weight)

        return self.to_cross_metric(tf_metric, array_type=type(y_true))

    def parse_sample_weight(self, sample_weight) -> Optional[tf.Tensor]:
        if sample_weight is None:
            return None

        return convert_array(sample_weight, tf.Tensor)


def to_tf_metric(
    to_convert: Union[CrossMetric, Aggregator], jit_compile=False
) -> tf.keras.metrics.Metric:
    if isinstance(to_convert, CrossMetric):
        return CrossMetricKeras(to_convert, jit_compile=jit_compile)
    return AggregatorMetric(to_convert)


class CrossTFMeanMetric(CrossTFMetricAggregator[TFMetricType, Mean]):
    def __init__(self, tf_metric: TFMetricType, **kwargs):
        self._tf_metric = tf_metric
        super().__init__(**kwargs)

    def tf_metric(self) -> tf.keras.metrics.Mean:
        return self._tf_metric

    def to_cross_metric(
        self,
        metric: tf.keras.metrics.Mean,
        array_type,
    ) -> Mean:
        return Mean(
            count=convert_array(tf.convert_to_tensor(metric.count), array_type),
            sum=convert_array(tf.convert_to_tensor(metric.total), array_type),
        )

    def present(self, state):
        return state.result


def from_tf_metric(
    metric: tf.keras.metrics.Metric, cross_metric_cls=CrossTFMeanMetric
) -> CrossTFMeanMetric:
    return cross_metric_cls(metric)
