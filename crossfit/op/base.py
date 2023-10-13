import inspect
from typing import Optional
import uuid

import dask.dataframe as dd
import pandas as pd
from dask.distributed import get_worker
from tqdm.auto import tqdm


class Op:
    def __init__(self, pre=None, cols=False, keep_cols=None, post=None):
        self.pre = pre
        self.post = post
        self.cols = cols
        self.keep_cols = keep_cols or []
        self.id = str(uuid.uuid4())
        self._input_type = None

    def setup(self):
        pass

    def meta(self):
        return None

    def setup_worker(self):
        try:
            worker = get_worker()
        except ValueError:
            worker = self

        self.worker_name = getattr(worker, "name", 0)
        init_name = f"setup_done_{self.id}"

        if not hasattr(worker, init_name):
            self.setup()
            setattr(worker, init_name, True)

    def call_dask(self, data: dd.DataFrame):
        output = data.map_partitions(self, meta=self._build_dask_meta(data))

        return output

    def create_df(self, *args, **kwargs):
        return self._input_type(*args, **kwargs)

    def create_series(self, *args, **kwargs):
        if isinstance(self._input_type, pd.DataFrame):
            return pd.Series(*args, **kwargs)

        import cudf

        return cudf.Series(*args, **kwargs)

    def create_progress_bar(self, total, partition_info=None, **kwargs):
        return tqdm(
            total=total,
            position=int(self.worker_name),
            desc=f"GPU: {self.worker_name}, Part: {partition_info['number']}",
            **kwargs,
        )

    def add_keep_cols(self, data, output):
        if not self.keep_cols:
            return output

        for col in self.keep_cols:
            if col not in output.columns:
                output[col] = data[col]

        columns = list(output.columns)
        output = output[self.keep_cols + columns]

        return output

    def __call__(self, data, *args, partition_info=None, **kwargs):
        if isinstance(data, dd.DataFrame):
            return self.call_dask(data, *args, **kwargs)

        self.setup_worker()

        inputs = data
        self._input_type = type(inputs)
        if self.pre is not None:
            params = inspect.signature(self.pre).parameters
            if "partition_info" in params:
                inputs = self.pre(inputs, partition_info=partition_info)
            else:
                inputs = self.pre(inputs)

        params = inspect.signature(self.call).parameters
        if "partition_info" in params:
            output = self.call(inputs, *args, partition_info=partition_info, **kwargs)
        else:
            output = self.call(inputs, *args, **kwargs)

        if self.post is not None:
            params = inspect.signature(self.post).parameters
            if "partition_info" in params:
                output = self.post(output, partition_info=partition_info)
            else:
                output = self.post(output)

        if self.keep_cols:
            output = self.add_keep_cols(data, output)

        return output

    def _build_dask_meta(self, data):
        output = {col: data[col].dtype for col in self.keep_cols}
        meta = self.meta()
        if meta:
            output.update(meta)

        return output


class ColumnOp(Op):
    def __init__(
        self,
        input_col: str,
        dtype: str,
        output_col: Optional[str] = None,
        keep_cols=None,
    ):
        super().__init__(
            pre=self.get_col, post=self.produce_output, keep_cols=keep_cols
        )
        self.input_col = input_col
        self.output_col = output_col or input_col
        self.dtype = dtype

    def get_col(self, data):
        return data[self.input_col]

    def produce_output(self, col):
        import cudf

        # TODO: Add support for pandas
        df = cudf.DataFrame()
        df[self.output_col] = col

        return df

    def _build_dask_meta(self, data):
        output = {col: data[col].dtype for col in self.keep_cols}
        meta = self.meta()

        if not isinstance(meta, dict):
            meta = {self.output_col: meta}

        if meta:
            output.update(meta)

        return output

    def meta(self):
        return self.dtype


class Repartition(Op):
    def __init__(self, partition_size: int, min_paritions=2):
        super().__init__()
        self.partition_size = partition_size
        self.min_paritions = min_paritions

    def call(self, data):
        return data

    def call_dask(self, data):
        partitions = max(int(len(data) / self.partition_size), 1)
        if partitions < self.min_paritions:
            partitions = self.min_paritions

        return data.repartition(partitions)
