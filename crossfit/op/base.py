import inspect
from typing import Dict, List, Generic, Union, TypeVar
import uuid
from typing_utils import get_args

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

    def _call_df(self, data, *args, partition_info=None, **kwargs):
        params = inspect.signature(self.call).parameters
        if "partition_info" in params:
            output = self.call(data, *args, partition_info=partition_info, **kwargs)
        else:
            output = self.call(data, *args, **kwargs)

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

        output = self._call_df(inputs, *args, partition_info=partition_info, **kwargs)
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


DtypeT = TypeVar("DtypeT")


class ColumnOp(Op, Generic[DtypeT]):
    def __init__(
        self,
        cols: Union[str, List[str], Dict[str, str]],
        output_dtype: DtypeT = None,
        keep_cols=None,
    ):
        super().__init__(keep_cols=keep_cols)
        self.cols = [cols] if isinstance(cols, str) else cols

        if output_dtype is None:
            # Infer output dtype from generic
            generics = get_args(self.__orig_bases__[0])
            if isinstance(generics, tuple):
                output_dtype = generics[0].__name__
                if output_dtype == "float":
                    output_dtype = "float32"
                if output_dtype == "int":
                    output_dtype = "int32"
            else:
                raise ValueError("Could not infer output_dtype, please specify it.")

        self.output_dtype = output_dtype

    def _call_df(self, data, *args, partition_info=None, **kwargs):
        output = self.create_df()

        if self.cols is None:
            if not str(type(data)).endswith("Series"):
                raise ValueError("data must be a Series")

            return self.call(data, *args, partition_info=partition_info, **kwargs)

        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")

            col_out = self.call(data[col])
            output[self._construct_name(col)] = col_out

        return output

    def _construct_name(self, col_name):
        if isinstance(self.cols, dict):
            return self.cols[col_name]

        return col_name

    def meta(self):
        if not self.cols:
            return self.output_dtype

        if isinstance(self.cols, dict):
            return {self.cols[col]: self.output_dtype for col in self.cols}

        return {col: self.output_dtype for col in self.cols}


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
