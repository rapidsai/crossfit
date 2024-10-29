# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import dask.dataframe as dd
from dask.distributed import get_worker, wait
from tqdm.auto import tqdm

from crossfit.backend.dask.cluster import global_dask_client


class Op:
    def __init__(self, pre=None, cols=False, keep_cols=None):
        self.pre = pre
        self.cols = cols
        self.keep_cols = keep_cols or []

    @property
    def worker_name(self):
        return getattr(self.get_worker(), "worker_address")

    def setup(self):
        pass

    def teardown(self):
        pass

    def meta(self):
        return None

    def get_worker(self):
        try:
            worker = get_worker()
        except ValueError:
            worker = self

        return worker

    def call_dask(self, data: dd.DataFrame):
        output = data.map_partitions(self, meta=self._build_dask_meta(data))

        if global_dask_client():
            wait(output)

        return output

    def create_progress_bar(self, total, partition_info=None, **kwargs):
        return tqdm(
            total=total,
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
        # we use dict.fromkeys to remove duplicates and preserve order
        # (to match _build_dask_meta behavior)
        output = output[list(dict.fromkeys(self.keep_cols + columns))]

        return output

    def __call__(self, data, *args, partition_info=None, **kwargs):
        if isinstance(data, dd.DataFrame):
            output = self.call_dask(data, *args, **kwargs)
            self.teardown()
            return output

        self.setup()

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
