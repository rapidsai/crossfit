from typing import List, Union

import cudf

from crossfit.op.base import Op


class Labeler(Op):
    def __init__(
        self,
        labels: List[str],
        cols=None,
        keep_cols=None,
        pre=None,
        keep_prob: bool = False,
        suffix: str = "labels",
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.labels = labels
        self.keep_prob = keep_prob
        self.suffix = suffix

    def call_column(self, data: cudf.Series) -> cudf.Series:
        if isinstance(data, cudf.DataFrame):
            raise ValueError(
                "data must be a Series, got DataFrame. Add a pre step to convert to Series"
            )

        num_labels = len(data.iloc[0])
        if len(self.labels) != num_labels:
            raise ValueError(
                f"The number of provided labels is {len(self.labels)} "
                f"but there are {num_labels} in data."
            )

        scores = data.list.leaves.values.reshape(-1, num_labels)
        classes = scores.argmax(1)
        labels_map = {i: self.labels[i] for i in range(len(self.labels))}

        return cudf.Series(classes).map(labels_map)

    def call(self, data: Union[cudf.Series, cudf.DataFrame]) -> Union[cudf.Series, cudf.DataFrame]:
        output = cudf.DataFrame()

        if self.cols is None:
            if not isinstance(data, cudf.Series):
                raise ValueError("data must be a cudf Series")

            return self.call_column(data)

        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")

            labels = self.call_column(data[col])
            output[self._construct_name(col, self.suffix)] = labels

        return output

    def meta(self):
        labeled = {"labels": "string"}

        if len(self.cols) > 1:
            labeled = {
                self._construct_name(col, suffix): dtype
                for col in self.cols
                for suffix, dtype in labeled.items()
            }

        return labeled

    def _construct_name(self, col_name, suffix):
        if len(self.cols) == 1:
            return suffix

        return f"{col_name}_{suffix}"
