from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
import crossfit as cf


def state(data, state_cls):
    args = {field.name: data[field.name] for field in fields(state_cls)}

    return state_cls(**args)


@dataclass
class DummyState(cf.MetricState):
    a: cf.Array = cf.field(combine=np.maximum)

    @property
    def result(self):
        return self.a * 2


@dataclass
class NoFields(cf.MetricState):
    a: cf.Array

    def combine(self, other):
        return replace(self, a=self.a + other.a)

    @property
    def result(self):
        return self.a * 2


def test_state_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    s = state(df, DummyState)
    s2 = state(df, NoFields)
    np.testing.assert_array_equal(s.result, np.array([2, 4, 6]))
    np.testing.assert_array_equal(s2.result, np.array([2, 4, 6]))
