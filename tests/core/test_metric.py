from dataclasses import dataclass

import numpy as np

from crossfit.core.metric import Combiner, MetricState, State
from crossfit.dispatch.combiner import Min


@dataclass
class DummyState(MetricState):
    some_state: State[Min]


def test_metric_state_is_adder():
    dummy = DummyState(np.array([1, 2, 3]))

    assert isinstance(dummy, Combiner)
    added = dummy + dummy
    assert np.all(added.some_state == dummy.some_state)
