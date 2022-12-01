import numpy as np

import crossfit as cf

import pandas as pd

array = np.array(list(range(10)))
df = pd.DataFrame({"a": [1, 2, 3] * 2, "b": [4, 5, 6] * 2})


a = 5


def test_min():
    min = cf.stats.Min()
    min.update(array)
    assert min.result() == 0
    
    
def test_max():
    max = cf.stats.Max()
    max.update(array)
    assert max.result() == 9
