import gc

from dask.distributed import Client
from dask_cuda import LocalCUDACluster


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


def setup_dask_cluster(rmm_pool_size="14GB", CUDA_VISIBLE_DEVICES="0,1"):
    """
    This function sets up a dask cluster across n GPUs.
    It also ensures maximum memory efficiency for the GPU by:
        1. Ensuring pytorch memory pool is the same as the RAPIDS memory pool.
        2. enables spilling for cudf.

    Args:
        rmm_pool_size: The size of the RMM pool to be used by each worker.
        CUDA_VISIBLE_DEVICES: The GPUs to be used by the cluster.
    Returns:
        A dask client object.

    """
    if rmm_pool_size is None:
        rmm_pool_size = True
    cluster = LocalCUDACluster(
        rmm_pool_size=rmm_pool_size,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        # log_spilling=True,
        # device_memory_limit=0.7,
        jit_unspill=True,
    )
    client = Client(cluster)
    client.run(set_torch_to_use_rmm)
    client.run(increase_gc_threshold)
    client.run(enable_spilling)

    return client


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
