import time

import dask

from crossfit.core.frame import MetricFrame
from crossfit.dask.calculate import calculate_per_col as calculate_dask
from crossfit.stats.continuous.stats import ContinuousStats

from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster

# Benchmark assumes Criteo dataset.
# Low-cardinality columns:
# {C6:4, C9:64, C13:11, C16:155, C17:4, C19:15, C25:109, C26:37}

# Options
path = "/raid/dask-space/criteo/crit_pq_int"
backend = "cudf"
split_row_groups = 10
ncolumns = 10
groupby = None
use_cluster = True


# Set Dask backend
dask.config.set({"dataframe.backend": backend})
if backend == "cudf":
    # For older dask versions, backend config wont work
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
        split_row_groups=split_row_groups,
        columns=columns,
    )
    print(f"\nddf: {ddf}\n")

    # Calculate continuous stats
    metric = ContinuousStats()
    t0 = time.time()
    mf: MetricFrame = calculate_dask(metric, ddf, groupby=groupby)
    tf = time.time()
    print(f"\nWall Time: {tf-t0} seconds\n")

    # View result
    assert isinstance(mf, MetricFrame)
    result = mf.result()
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
