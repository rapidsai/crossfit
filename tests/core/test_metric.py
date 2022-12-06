from dataclasses import dataclass

import numpy as np

import crossfit as cf
from crossfit.core.metric import Array, Combine
import crossfit.array.ops as cnp


@dataclass
class DummyState(cf.MetricState):
    some_state: cf.Array = cf.field(combine=np.maximum)


@dataclass
class StateWithChild(cf.MetricState):
    child: DummyState
    state: Array = cf.field(combine=cnp.maximum)


def test_metric_state_can_be_combined():
    dummy = DummyState(np.array([1, 2, 3]))

    assert isinstance(dummy, Combine)
    combined = dummy + dummy
    assert np.all(combined.some_state == dummy.some_state)


def test_metric_state_with_child_can_be_combined():
    with_child = StateWithChild(DummyState(np.array([1, 2, 3])), np.array([1, 2, 3]))

    assert isinstance(with_child, Combine)
    assert isinstance(with_child + with_child, StateWithChild)
