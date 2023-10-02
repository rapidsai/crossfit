from typing import List
import math

import cudf
import cupy as cp
from numba import cuda
import dask_cudf
from cuml.preprocessing import LabelEncoder

from crossfit.backend.dask.aggregate import aggregate
from crossfit.dataset.base import EmbeddingDatataset
from crossfit.report.beir.embed import embed
from crossfit.calculate.aggregate import Aggregator
from crossfit.metric.continuous.mean import Mean


def dcg_at_k(r, k):
    """Vectorized DCG@k calculation"""
    k_idx = cp.arange(1, k + 1)
    return cp.sum((cp.power(2, r[:, :k]) - 1) / cp.log2(k_idx + 1), axis=1)


@cuda.jit
def row_topk_csr_cuda(data, indices, indptr, max_indices, max_values, K):
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row >= indptr.shape[0] - 1:  # boundary check
        return

    start = indptr[row]
    end = indptr[row + 1]
    size = end - start

    # Shared memory for sorting
    shared_data = cuda.shared.array(shape=(1024,), dtype=cp.float32)
    shared_indices = cuda.shared.array(shape=(1024,), dtype=cp.int32)

    for i in range(size):
        shared_data[i] = data[start + i]
        shared_indices[i] = indices[start + i]

    cuda.syncthreads()

    # Sorting (could be optimized)
    for i in range(size):
        for j in range(i + 1, size):
            if shared_data[i] < shared_data[j]:
                shared_data[i], shared_data[j] = shared_data[j], shared_data[i]
                shared_indices[i], shared_indices[j] = shared_indices[j], shared_indices[i]

    cuda.syncthreads()

    # Write top K values and indices
    for i in range(K):
        if i < size:
            max_indices[row, i] = shared_indices[i]
            max_values[row, i] = shared_data[i]


def topk_csr(csr_matrix, K):
    # Initialize output arrays
    num_rows = csr_matrix.shape[0]
    max_indices = cp.zeros((num_rows, K), dtype=cp.int32)
    max_values = cp.zeros((num_rows, K), dtype=cp.float32)

    # CUDA kernel config
    threads_per_block = 32
    blocks = math.ceil(num_rows / threads_per_block)

    row_topk_csr_cuda[blocks, threads_per_block](
        csr_matrix.data, csr_matrix.indices, csr_matrix.indptr, max_indices, max_values, K
    )

    return max_indices, max_values


def ndcg_at_k(preds, true_relevance, k, sorted_pred_idx=None, to_dense=False):
    # Step 1: Sort predicted indices
    if sorted_pred_idx is None:
        sorted_pred_idx = cp.argsort(-preds, axis=1)

    sorted_idx = sorted_pred_idx[:, :k]

    # Step 2: Fetch the true relevance scores using sorted indices
    rows = cp.repeat(cp.arange(preds.shape[0]), k)
    sorted_true_relevance = true_relevance[rows, sorted_idx.flatten()].reshape(preds.shape[0], k)

    # Step 3: Calculate DCG
    dcg_values = dcg_at_k(sorted_true_relevance, k)

    # Step 4: Calculate IDCG
    if to_dense:
        sorted_true_relevance_best = -cp.sort(-true_relevance.toarray(), axis=1)[:, :k]
        idcg_values = dcg_at_k(sorted_true_relevance_best, k)
    else:
        _, top_val = topk_csr(true_relevance, k)
        idcg_values = dcg_at_k(top_val, k)

    # Replace 0s with 1s to avoid division by 0
    idcg_values[idcg_values == 0] = 1.0

    # Step 5: Calculate NDCG
    ndcg_values = dcg_values / idcg_values

    return ndcg_values


class BeirMetricAggregator(Aggregator):
    def __init__(
        self,
        ks: List[int],
        pre=None,
        post_group=None,
        post=None,
        groupby=None,
    ):
        super().__init__(None, pre=pre, post_group=post_group, post=post, groupby=groupby)
        self.ks = ks

    def prepare(self, df):
        encoder = self.create_label_encoder(df, ["corpus-index-pred", "corpus-index-obs"])
        obs_csr = self.create_csr_matrix(df["corpus-index-obs"], df["score-obs"], encoder)

        max_k = len(df["corpus-index-pred"].iloc[0])
        pred_indices = encoder.transform(df["corpus-index-pred"].list.leaves).values.reshape(
            -1, max_k
        )
        pred_scores = df["score-pred"].list.leaves.values.reshape(-1, max_k)

        outputs = {}

        for k in self.ks:
            m = ndcg_at_k(pred_scores, obs_csr, k=k, sorted_pred_idx=pred_indices, to_dense=True)
            outputs[f"ndcg-at-{k}"] = Mean.from_array(m, axis=0)

        return outputs

    def create_label_encoder(self, df, cols) -> LabelEncoder:
        # Extract leaves (flattened arrays)
        _leaves = []

        for col in cols:
            _leaves.append(df[col].list.leaves)

        # Concatenate and get unique values for fit_transform
        all_ids = cudf.concat(_leaves).unique()

        # Label Encoding
        le = LabelEncoder()
        le.fit(all_ids)

        return le

    def create_csr_matrix(self, ids, scores, label_encoder: LabelEncoder):
        num_rows = scores.size
        num_columns = label_encoder.classes_.shape[0]

        values = scores.list.leaves.values.astype(cp.float32)
        indices = label_encoder.transform(ids.list.leaves).values
        indptr = scores.list._column.offsets.values
        sparse_matrix = cp.sparse.csr_matrix(
            (values, indices, indptr), shape=(num_rows, num_columns)
        )

        return sparse_matrix


def join_predictions(data, predictions):
    print("Joining predictions...")

    if hasattr(predictions, "ddf"):
        predictions = predictions.ddf()

    if hasattr(predictions, "ddf"):
        data = data.ddf()

    observed = (
        data[["query-index", "corpus-index", "score", "split"]]
        .groupby("query-index")
        .agg(
            {"corpus-index": list, "score": list, "split": "first"},
            split_out=data.npartitions,
            shuffle=True,
        )
    )

    predictions = predictions.set_index("query-index")
    merged = observed.merge(
        predictions, left_index=True, right_index=True, how="left", suffixes=("-obs", "-pred")
    ).rename(columns={"split-obs": "split"})

    output = merged.reset_index()

    return output


def beir_report(
    dataset_name: str,
    model_name: str,
    partition_num: int = 50_000,
    ks=[1, 3, 5, 10],
    overwrite=False,
    out_dir=None,
    client=None,
    groupby=["split"],
):
    embeddings: EmbeddingDatataset = embed(
        dataset_name,
        model_name=model_name,
        partition_num=partition_num,
        overwrite=overwrite,
        out_dir=out_dir,
        client=client,
        dense_search=True,
    )

    observations = []
    for split in ["train", "val", "test"]:
        split_data = getattr(embeddings.data, split)

        if split_data is None:
            continue

        ddf = split_data.ddf()
        ddf["split"] = split

        observations.append(ddf)

    data = dask_cudf.concat(observations)
    joined = join_predictions(data, embeddings.predictions)

    # partitions = max(int(len(joined) / partition_num / 10), 1)
    # if not partitions % 2 == 0:
    #     partitions += 1

    joined = joined.repartition(20)

    aggregator = BeirMetricAggregator(ks)
    aggregator = Aggregator(aggregator, groupby=groupby, name="")

    results = aggregate(joined, aggregator, to_frame=True)

    return results
