from typing import List

import cudf
import cupy as cp
from numba import cuda
import dask
from cuml.preprocessing import LabelEncoder

from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.dataset.base import EmbeddingDatataset, IRData
from crossfit.report.beir.embed import embed
from crossfit.calculate.aggregate import Aggregator


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
        encoder = self.create_label_encoder(df, ["corpus-id-pred", "corpus-id-obs"])
        obs_csr = self.create_csr_matrix(df["corpus-id-obs"], df["score-obs"], encoder)

        max_k = len(df["corpus-id-pred"].iloc[0])
        pred_indices = encoder.transform(df["corpus-id-pred"].list.leaves).values.reshape(-1, max_k)
        pred_scores = df["score-pred"].list.leaves.values.reshape(-1, max_k)

        outputs = {}

        for k in self.ks:
            m = ndcg_at_k(pred_scores, obs_csr, k=k, sorted_pred_idx=pred_indices, to_dense=True)
            outputs[f"ndcg-at-{k}"] = cf.Mean.from_array(m, axis=0)

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


def beir_report(
    dataset_name: str,
    model_name: str,
    partition_num: int = 50_000,
    ks=[1, 3, 5, 10],
    split="test",
    overwrite=False,
    out_dir=None,
    client=None,
    groupby=None,
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

    if not hasattr(embeddings.data, split):
        raise ValueError(f"Dataset {dataset_name} does not have split {split}")

    data: IRData = getattr(embeddings.data, split)
    joined = data.join_predictions(embeddings.predictions).repartition(10)

    # aggregator = BeirMetricAggregator(ks)
    # results = CrossFrame(joined).aggregate(aggregator, to_frame=True)

    # return results

    return joined
