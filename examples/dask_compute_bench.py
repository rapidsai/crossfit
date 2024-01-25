# Copyright 2023-2024 NVIDIA CORPORATION
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

import time

import dask
from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster

import crossfit as cf
from crossfit.backend.dask.aggregate import aggregate
from crossfit.metric.continuous.moments import Moments

# Benchmark assumes Criteo dataset.
# Low-cardinality columns:
# {C6:4, C9:64, C13:11, C16:155, C17:4, C19:15, C25:109, C26:37}

# Options
path = "/datasets/rzamora/crit_pq_int"
backend = "cudf"
blocksize = "265MiB"
ncolumns = 4
groupby = None
use_cluster = True


# Set Dask backend
dask.config.set({"dataframe.backend": backend})
if backend == "cudf":
    # For older dask versions, backend config won't work
    import dask_cudf as dd
else:
    import dask.dataframe as dd

if __name__ == "__main__":
    if use_cluster:
        # Spin up cluster
        cluster_type = LocalCUDACluster if backend == "cudf" else LocalCluster
        client = Client(cluster_type())

    # Define DataFrame collection
    columns = [f"I{i}" for i in range(1, ncolumns + 1)]
    if groupby:
        columns += groupby if isinstance(groupby, list) else [groupby]
    ddf = dd.read_parquet(
        path,
        blocksize=blocksize,
        columns=columns,
    )
    print(f"\nddf: {ddf}\n")

    # Calculate continuous stats
    agg = cf.Aggregator(Moments(axis=0), per_column=True)
    t0 = time.time()
    result = aggregate(ddf, agg, to_frame=True)
    tf = time.time()
    print(f"\nWall Time: {tf-t0} seconds\n")

    # View result
    print(f"Result:\n{result}\n")
    print(f"Type: {type(result)}\n")

    if groupby:
        # Compare to ddf.groupby(groupby).std()
        t0 = time.time()
        std = ddf.groupby(groupby).std().compute()
        tf = time.time()
        print(f"\nddf.groupby().std() takes {tf-t0} seconds, and returns:\n")
        print(f"\n{std}\n")
    else:
        # Compare to ddf.std()
        t0 = time.time()
        std = ddf.std().compute()
        tf = time.time()
        print(f"\nddf.std() takes {tf-t0} seconds, and returns:\n")
        print(f"\n{std}\n")
