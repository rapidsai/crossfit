import inspect
import uuid

import dask.dataframe as dd
from tqdm import tqdm
from dask.distributed import get_worker


class Op:
    def __init__(self, pre=None, cols=False, keep_cols=None):
        self.pre = pre
        self.cols = cols
        self.keep_cols = keep_cols or []
        self.id = str(uuid.uuid4())

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

        if self.pre is not None:
            params = inspect.signature(self.pre).parameters
            if "partition_info" in params:
                data = self.pre(data, partition_info=partition_info)
            else:
                data = self.pre(data)

        params = inspect.signature(self.call).parameters
        if "partition_info" in params:
            output = self.call(data, *args, partition_info=partition_info, **kwargs)
        else:
            output = self.call(data, *args, **kwargs)

        if self.keep_cols:
            output = self.add_keep_cols(data, output)

        return output

    def _build_dask_meta(self, data):
        output = {col: data[col].dtype for col in self.keep_cols}
        meta = self.meta()
        if meta:
            output.update(meta)

        return output
