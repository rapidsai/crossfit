import inspect
import uuid

from dask.distributed import get_worker
import dask.dataframe as dd


class Op:
    def __init__(
        self,
        pre=None,
        cols=False,
    ):
        self.pre = pre
        self.cols = cols
        self.id = str(uuid.uuid4())

    def setup(self):
        pass

    def meta(self):
        return None

    def setup_worker(self):
        worker = get_worker() or self

        self.worker_name = getattr(worker, "name", "local")
        init_name = f"setup_done_{self.id}"

        if not hasattr(worker, init_name):
            self.setup()
            setattr(worker, init_name, True)

    def call_dask(self, data):
        return data.map_partitions(self, meta=self.meta())

    def __call__(self, data, partition_info=None):
        if isinstance(data, dd.DataFrame):
            return self.call_dask(data)

        self.setup_worker()

        if self.pre is not None:
            params = inspect.signature(self.pre).parameters
            if "partition_info" in params:
                data = self.pre(data, partition_info=partition_info)
            else:
                data = self.pre(data)

        params = inspect.signature(self.call).parameters
        if "partition_info" in params:
            return self.call(data, partition_info=partition_info)
        else:
            return self.call(data)
