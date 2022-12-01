from typing import Optional

from dataclasses import dataclass
from crossfit.core.frame import DataClassFrame
import pandas as pd


df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@dataclass
class Mean:
    summed: int
    counted: int
    
    index: Optional[str] = None
    
    @property
    def name(self):
        if self.index is not None:
            return self.index
        
        return self.__class__.__name__
    
    def result(self, state):
        return state.summed / state.counted


records = [
    Mean(10, 1, index="col1"),
    Mean(11, 2, index="col2"),
    Mean(12, 3, index="col3"),
]

dcf = DataClassFrame(
    record_class=Mean,
    data=records,
    index=["name"]
)

a = 5