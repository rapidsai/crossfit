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

from typing import List, Optional

import cudf
import cupy as cp
import dask_cudf
from cuml.preprocessing import LabelEncoder

from crossfit.backend.dask.aggregate import aggregate
from crossfit.backend.torch.loader import DEFAULT_BATCH_SIZE
from crossfit.backend.torch.model import Model
from crossfit.calculate.aggregate import Aggregator
from crossfit.data.array.dispatch import crossarray
from crossfit.dataset.base import EmbeddingDatataset
from crossfit.metric.continuous.mean import Mean
from crossfit.metric.ranking import AP, NDCG, Precision, Recall, SparseNumericLabels, SparseRankings
from crossfit.op.vector_search import VectorSearchOp
from crossfit.report.base import Report
from crossfit.report.beir.embed import embed
from crossfit.utils.torch_utils import cleanup_torch_cache


class BeirMetricAggregator(Aggregator):
    def __init__(
        self,
        ks: List[int],
        pre=None,
        post_group=None,
        post=None,
        groupby=None,
        metrics=[NDCG, AP, Precision, Recall],
    ):
        super().__init__(None, pre=pre, post_group=post_group, post=post, groupby=groupby)
        self.ks = ks
        self.metrics = metrics

    def prepare(self, df):
        encoder = create_label_encoder(df, ["corpus-index-pred", "corpus-index-obs"])
        obs_csr = create_csr_matrix(df["corpus-index-obs"], df["score-obs"], encoder)
        pred_csr = create_csr_matrix(df["corpus-index-pred"], df["score-pred"], encoder)

        # TODO: Fix dispatch
        labels = SparseNumericLabels.from_matrix(obs_csr)
        rankings = SparseRankings.from_scores(pred_csr)

        outputs = {}
        with crossarray:
            for metric in self.metrics:
                for k in self.ks:
                    metric_at_k = metric(k=k)
                    result = metric_at_k.score(labels, rankings)

                    outputs[metric_at_k.name()] = Mean.from_array(result, axis=0)

        return outputs


def create_label_encoder(df, cols) -> LabelEncoder:
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


def create_csr_matrix(ids, scores, label_encoder: LabelEncoder):
    num_rows = scores.size
    num_columns = label_encoder.classes_.shape[0]

    values = scores.list.leaves.values.astype(cp.float32)
    indices = label_encoder.transform(ids.list.leaves).values
    indptr = scores.list._column.offsets.values
    sparse_matrix = cp.sparse.csr_matrix((values, indices, indptr), shape=(num_rows, num_columns))

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
        predictions,
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("-obs", "-pred"),
    ).rename(columns={"split-obs": "split"})

    output = merged.reset_index()

    return output


class BeirReport(Report):
    def __init__(self, result_df):
        self.result_df = result_df

    def visualize(self, name="data"):
        raise NotImplementedError()

    def console(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print(self.result_df)

        for i in range(len(self.result_df)):
            console.rule(": ".join(self.result_df.index[i]))
            grouped_columns = {}
            for col in self.result_df.columns:
                metric_type = col.split("@")[0] if "@" in col else col
                grouped_columns.setdefault(metric_type, []).append(col)

            # Sort the @k values within each group
            for metric, columns in grouped_columns.items():
                grouped_columns[metric] = sorted(columns, key=lambda x: int(x.split("@")[-1]))

            # Print table for each metric type
            for metric, columns in grouped_columns.items():
                table = Table(show_header=True, header_style="bold magenta")
                for column_name in columns:
                    table.add_column(column_name)

                row_data = self.result_df.iloc[i][columns]
                table.add_row(*[str(row_data[col]) for col in columns])

                console.print(table)


def beir_report(
    dataset_name: str,
    model: Model,
    vector_search: VectorSearchOp,
    partition_num: Optional[int] = 50_000,
    ks=[1, 3, 5, 10],
    overwrite=False,
    out_dir=None,
    groupby=["split"],
    tiny_sample=False,
    sorted_data_loader: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BeirReport:
    embeddings: EmbeddingDatataset = embed(
        dataset_name,
        model=model,
        partition_num=partition_num,
        overwrite=overwrite,
        out_dir=out_dir,
        vector_search=vector_search,
        tiny_sample=tiny_sample,
        sorted_data_loader=sorted_data_loader,
        batch_size=batch_size,
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

    del data
    del embeddings
    cleanup_torch_cache()
    aggregator = BeirMetricAggregator(ks)
    aggregator = Aggregator(aggregator, groupby=groupby, name="")

    results = aggregate(joined, aggregator, to_frame=True)

    return BeirReport(results)
