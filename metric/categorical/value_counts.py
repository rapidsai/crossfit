import pandas as pd
import numpy as np

from crossfit.metric.base import CrossMetric, state


class ValueCounts(CrossMetric):
    values = state(init=np.array([]), is_list=True)
    counts = state(init=np.array([]), is_list=True)

    def __init__(self, k=50, values=None, counts=None):
        self.k = k
        self.setup(values=values, counts=counts)

    def prepare(self, data, axis=0):
        if hasattr(data, "value_counts"):
            d = data.value_counts().sort_index()
            values = d.index
            counts = d.values
        else:
            values, counts = np.unique(data, axis=axis, return_counts=True)

        return ValueCounts(values=values, counts=counts)

    @property
    def as_frame(self):
        data = {"value": self.values, "count": self.counts}
        if hasattr(self.values, "to_pandas"):
            import cudf as cd

            return cd.DataFrame(data).set_index("value")

        elif not isinstance(self.values, (np.ndarray, pd.Index, pd.Series)):
            raise NotImplementedError("Only numpy & cupy arrays are supported for now")

        return pd.DataFrame(data).set_index("value")

    def combine(self, other) -> "ValueCounts":
        self_frame = self.as_frame
        other_frame = other.as_frame

        # TODO: Implement this properly
        #   It seems that cudf doesn't support using lsuffix & rsuffix
        if hasattr(self_frame, "to_pandas"):
            self_frame = self_frame.to_pandas()
        if hasattr(other_frame, "to_pandas"):
            other_frame = other_frame.to_pandas()

        combined_frame = self_frame.join(
            other_frame,
            lsuffix="_l",
            rsuffix="_r",
            how="outer",
        ).fillna(0)
        combined = (combined_frame["count_l"] + combined_frame["count_r"]).astype("int64")

        return ValueCounts(values=combined.index._data, counts=combined.values)

    def top_k(self, k: int) -> pd.DataFrame:
        largest = self.as_frame.nlargest(k, "count")
        if hasattr(self.values, "to_pandas"):
            return largest.to_pandas()

        return largest

    def present(self):
        top_k = self.top_k(self.k)

        return {
            "num_unique": len(self.values),
            "top_counts": top_k["count"].values,
            "top_values": top_k["count"].index._data,
        }
