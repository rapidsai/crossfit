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

import gc
import importlib
import warnings
from contextvars import ContextVar
from typing import Any, Callable, Optional

import dask
import distributed
from dask.dataframe.optimize import optimize as dd_optimize
from dask.distributed import Client, get_client

from crossfit.backend.gpu import HAS_GPU

_crossfit_dask_client = ContextVar("_crossfit_dask_client", default="auto")


def set_torch_to_use_rmm():
    """
    This function sets up the pytorch memory pool to be the same as the RAPIDS memory pool.
    This helps avoid OOM errors when using both pytorch and RAPIDS on the same GPU.
    See article:
    https://medium.com/rapids-ai/pytorch-rapids-rmm-maximize-the-memory-efficiency-of-your-workflows-f475107ba4d4
    """
    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


# def setup_dask_cluster(rmm_pool_size="14GB", CUDA_VISIBLE_DEVICES="0,1", jit_unspill=True):
#     """
#     This function sets up a dask cluster across n GPUs.
#     It also ensures maximum memory efficiency for the GPU by:
#         1. Ensuring pytorch memory pool is the same as the RAPIDS memory pool.
#         2. enables spilling for cudf.

#     Args:
#         rmm_pool_size: The size of the RMM pool to be used by each worker.
#         CUDA_VISIBLE_DEVICES: The GPUs to be used by the cluster.
#     Returns:
#         A dask client object.

#     """
#     from dask_cuda import LocalCUDACluster


#     if rmm_pool_size is None:
#         rmm_pool_size = True
#     cluster = LocalCUDACluster(
#         rmm_pool_size=rmm_pool_size,
#         CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
#         # log_spilling=True,
#         # device_memory_limit=0.7,
#         jit_unspill=jit_unspill,
#     )
#     client = Client(cluster)
#     client.run(set_torch_to_use_rmm)
#     client.run(increase_gc_threshold)
#     if jit_unspill:
#         client.run(enable_spilling)

#     return client


def enable_spilling():
    import cudf

    cudf.set_option("spill", True)


def increase_gc_threshold():
    # Trying to increase gc threshold to get rid of the warnings
    # This is due to this issue
    # in Sentence Transformers
    # See issue:
    # https://github.com/UKPLab/sentence-transformers/issues/487

    g0, g1, g2 = gc.get_threshold()
    gc.set_threshold(g0 * 3, g1 * 3, g2 * 3)


def ensure_optimize_dataframe_graph(ddf=None, dsk=None, keys=None):
    """Perform HLG DataFrame optimizations

    If `ddf` is specified, an optimized Dataframe
    collection will be returned. If `dsk` and `keys`
    are specified, an optimized graph will be returned.

    These optimizations are performed automatically
    when a DataFrame collection is computed/persisted,
    but they are NOT always performed when statistics
    are computed. The purpose of this utility is to
    ensure that the Dataframe-based optimizations are
    always applied.

    Parameters
    ----------
    ddf : dask_cudf.DataFrame, optional
        The dataframe to optimize, by default None
    dsk : dask.highlevelgraph.HighLevelGraph, optional
        Dask high level graph, by default None
    keys : List[str], optional
        The keys to optimize, by default None

    Returns
    -------
    Union[dask_cudf.DataFrame, dask.highlevelgraph.HighLevelGraph]
        A dask_cudf DataFrame or dask HighLevelGraph depending
        on the parameters provided.

    Raises
    ------
    ValueError
        If ddf is not provided and one of dsk or keys are None.
    """

    if ddf is None:
        if dsk is None or keys is None:
            raise ValueError("Must specify both `dsk` and `keys` if `ddf` is not supplied.")
    dsk = ddf.dask if dsk is None else dsk
    keys = ddf.__dask_keys__() if keys is None else keys

    if isinstance(dsk, dask.highlevelgraph.HighLevelGraph):
        with dask.config.set({"optimization.fuse.active": False}):
            dsk = dd_optimize(dsk, keys=keys)

    if ddf is None:
        # Return optimized graph
        return dsk

    # Return optimized ddf
    ddf.dask = dsk
    return ddf


class Distributed:
    """Distributed-Execution Context Manager

    The purpose of this execution-manager utility is to
    provide an intuitive context manager for distributed
    (multi-GPU/CPU) scheduling and execution with Dask.

    NOTE: For multi-node execution, it is the users
    responsibility to create a distributed Dask cluster
    with an appropriate deployment technology.  This
    class only supports the automatic generation of
    local (single-machine) clusters. However, the class
    can be used to connect to any existing dask cluster
    (local or not), as long as a valid `client` argument
    can be defined.

    Parameters
    -----------
    client : `dask.distributed.Client`; Optional
        The client to use for distributed-Dask execution.
    cluster_type : {"cuda", "cpu", None}
        Type of local cluster to generate in the case that a
        global client is not detected (or `force_new=True`).
        "cuda" corresponds to `dask_cuda.LocalCUDACluster`,
        while "cpu" corresponds to `distributed.LocalCluster`.
        Default is "cuda" if GPU support is detected.
    force_new : bool
        Whether to force the creation of a new local cluster
        in the case that a global client object is already
        detected. Default is False.
    **cluster_options :
        Key-word arguments to pass to the local-cluster
        constructor specified by `cluster_type` (e.g.
        `n_workers=2`).

    Examples
    --------
    The easiest way to use `Distributed` is within a
    conventional `with` statement::

        from merlin.core.utils import Disrtibuted

        workflow = nvt.Workflow(["col"] >> ops.Normalize())
        dataset = nvt.Dataset(...)
        with Distributed():
            workflow.fit(dataset)
            workflow.transform(dataset).to_parquet(...)

    In this case, all Dask-based scheduling and execution
    required within the `with Distributed()` block will be
    performed using a distributed cluster. If an existing
    client is not detected, a default `LocalCUDACluster`
    or `LocalCluster` will be automatically deployed (the
    specific type depending on GPU support).

    Alternatively, the distributed-execution manager can be
    used without a `with` statement as follows::

        workflow = nvt.Workflow(["col"] >> ops.Normalize())
        dataset = nvt.Dataset(...)
        exec = Distributed()
        workflow.fit(dataset)
        workflow.transform(dataset).to_parquet(...)
        exec.deactivate()

    Note that `deactivate()` must be used to resume default
    execution in the case that `Distributed` is not used in
    a `with` context.

    Since the default local cluster may be inefficient for
    many workflows, the user can also specify the specific
    `cluster_type` and `**cluster_options`. For example::

        with Distributed(
            cluster_type="cuda",
            force_new=True,  # Ignore existing cluster(s)
            n_workers=4,
            local_directory="/raid/dask-space",
            protocol="ucx",
            device_memory_limit=0.8,
            rmm_pool_size="20GB",
            log_spilling=True,
        ):
            workflow.fit(dataset)
            workflow.transform(dataset).to_parquet(...)

    In this case, the `cluster_type="cuda"` calls for the
    creation of a `LocalCUDACluster`, and all other key-word
    arguments are passed to the `LocalCUDACluster` constructor.
    """

    def __init__(
        self,
        client=None,
        cluster_type=None,
        force_new=False,
        enable_cudf_spilling=False,
        torch_rmm=True,
        increase_gc_threshold=True,
        **cluster_options,
    ):
        self._initial_client = global_dask_client()  # Initial state
        self._client = client or "auto"  # Cannot be `None`
        self.cluster_type = cluster_type or ("cuda" if HAS_GPU else "cpu")

        if torch_rmm and "rmm_pool_size" not in cluster_options:
            cluster_options["rmm_pool_size"] = True

        self.cluster_options = cluster_options
        # We can only shut down the cluster in `shutdown`/`__exit__`
        # if we are generating it internally
        set_dask_client(self._client)
        self._allow_shutdown = global_dask_client() is None or force_new
        self._active = False
        self.force_new = force_new
        self.enable_cudf_spilling = enable_cudf_spilling
        self.torch_rmm = torch_rmm
        self.increase_gc_threshold = increase_gc_threshold
        # Activate/deploy the client/cluster
        self._activate()

    @property
    def client(self):
        return self._client

    @property
    def cluster(self):
        return self.client.cluster

    @property
    def dashboard_link(self):
        return self.client.dashboard_link

    def _activate(self):
        if not self._active:
            self._client = set_dask_client(
                self._client,
                new_cluster=self.cluster_type,
                force_new=self.force_new,
                **self.cluster_options,
            )

            if self.torch_rmm:
                self.client.run(set_torch_to_use_rmm)
            if self.enable_cudf_spilling:
                self.client.run(enable_spilling)
            if self.increase_gc_threshold:
                self.client.run(increase_gc_threshold)

        self._active = True
        if self._client in ("auto", None):
            raise RuntimeError(f"Failed to deploy a new local {self.cluster_type} cluster.")

    def _deactivate(self):
        self._client = set_dask_client(self._initial_client)
        self._active = False

    def deactivate(self):
        if self._allow_shutdown and self._active:
            self._client.close()
        self._deactivate()

    def __enter__(self):
        self._activate()
        return self

    def __exit__(self, *args):
        self.deactivate()

    def __del__(self, *args):
        self.deactivate()


class Serial:
    """Serial-Execution Context Manager

    Examples
    --------
    The easiest way to use `Serial` is within a
    conventional `with` statement::

        from merlin.core.utils import Serial

        workflow = nvt.Workflow(["col"] >> ops.Normalize())
        dataset = nvt.Dataset(...)
        with Serial():
            workflow.transform(dataset).to_parquet(...)

    In this case, all Dask-based scheduling and execution
    required within the `with Serial()` block will be
    performed using the "synchronous" (single-threaded)
    scheduler.

    Alternatively, the serial-execution manager can be
    used without a `with` statement as follows::

        workflow = nvt.Workflow(["col"] >> ops.Normalize())
        dataset = nvt.Dataset(...)
        exec = Serial()
        workflow.transform(dataset).to_parquet(...)
        exec.deactivate()

    Note that `deactivate()` must be used to resume
    default execution in the case that `Serial` is
    not used in a `with` context.
    """

    def __init__(self):
        # Save the initial client setting and
        # activate serial-execution mode
        self._initial_client = global_dask_client()
        self._client = self._initial_client
        self._active = False
        self._activate()

    @property
    def client(self):
        return self._client

    def _activate(self):
        # Activate serial-execution mode.
        # This just means we are setting the
        # global dask client to `None`
        if not self._active:
            set_dask_client(None)
        self._active = True
        if global_dask_client() is not None:
            raise RuntimeError("Failed to activate serial-execution mode.")

    def deactivate(self):
        # Deactivate serial-execution mode.
        # This just means we are setting the
        # global dask client to the original setting.
        set_dask_client(self._initial_client)
        self._active = False
        if self._initial_client is not None and global_dask_client() is None:
            raise RuntimeError("Failed to deactivate serial-execution mode.")

    def __enter__(self):
        self._activate()
        return self

    def __exit__(self, *args):
        self.deactivate()


def set_dask_client(client="auto", new_cluster=None, force_new=False, **cluster_options):
    """Set the Dask-Distributed client.

    Parameters
    -----------
    client : {"auto", None} or `dask.distributed.Client`
        The client to use for distributed-Dask execution.
        If `"auto"` (default) the current python context will
        be searched for an existing client object. Specify
        `None` to disable distributed execution altogether.
    new_cluster : {"cuda", "cpu", None}
        Type of local cluster to generate in the case that
        `client="auto"` and a global dask client is not
        detected in the current python context. The "cuda"
        option corresponds to `dask_cuda.LocalCUDACluster`,
        while "cpu" corresponds to `distributed.LocalCluster`.
        Default is `None` (no local cluster is generated).
    force_new : bool
        Whether to force the creation of a new local cluster
        in the case that a global client object is already
        detected. Default is False.
    **cluster_options :
        Key-word arguments to pass to the local-cluster
        constructor specified by `new_cluster` (e.g.
        `n_workers=2`).
    """
    _crossfit_dask_client.set(client)

    # Check if we need to deploy a new cluster
    if new_cluster and client is not None:
        base, cluster = {
            "cuda": ("dask_cuda", "LocalCUDACluster"),
            "cpu": ("distributed", "LocalCluster"),
        }.get(new_cluster, (None, None))
        if global_dask_client() is not None and not force_new:
            # Don't deploy a new cluster if one already exists
            warnings.warn(
                "Existing Dask-client object detected in the "
                f"current context. New {new_cluster} cluster "
                "will not be deployed. Set force_new to True "
                "to ignore running clusters."
            )
        elif base and cluster:
            try:
                base = importlib.import_module(base)
            except ImportError as err:
                # ImportError should only occur for LocalCUDACluster,
                # but I'm making this general to be "safe"
                raise ImportError(
                    f"new_cluster={new_cluster} requires {base}. "
                    "Please make sure this library is installed. "
                ) from err

            cluster = getattr(base, cluster)(**cluster_options)
            print(f"Deployed {cluster}...")
            _crossfit_dask_client.set(Client(cluster))
        else:
            # Something other than "cuda" or "cpu" was specified
            raise ValueError(f"{new_cluster} not a supported option for new_cluster.")

    # Return the active client object
    active = _crossfit_dask_client.get()
    return None if active == "auto" else active


def global_dask_client() -> Optional[distributed.Client]:
    """Get Global Dask client if it's been set.

    Returns
    -------
    Optional[distributed.Client]
        The global client.
    """
    # First, check _merlin_dask_client
    crossfit_client = _crossfit_dask_client.get()
    if crossfit_client and crossfit_client != "auto":
        if crossfit_client.cluster and crossfit_client.cluster.workers:  # type: ignore
            # Active Dask client already known
            return crossfit_client
        else:
            # Our cached client is no-longer
            # active, reset to "auto"
            crossfit_client = "auto"
    if crossfit_client == "auto":
        try:
            # Check for a global Dask client
            set_dask_client(get_client())
            return _crossfit_dask_client.get()
        except ValueError:
            # no global client found
            pass
    # Catch-all
    return None


def run_on_worker(func: Callable, *args, **kwargs) -> Any:
    """Run a function on a Dask worker using `delayed`
    execution (if a Dask client is detected)

    Parameters
    ----------
    func : Callable
        The function to run

    Returns
    -------
    Any
        The result of the function call with supplied arguments
    """

    if global_dask_client():
        # There is a specified or global Dask client. Use it
        return dask.delayed(func)(*args, **kwargs).compute()
    # No Dask client - Use simple function call
    return func(*args, **kwargs)
