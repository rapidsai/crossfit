from typing import List
from crossfit.report.base import Report

import cudf
import cupy as cp
import dask_cudf
from cuml.preprocessing import LabelEncoder
import numpy as np

from crossfit.backend.dask.aggregate import aggregate
from crossfit.data.sparse.dispatch import CrossSparse
from crossfit.data.array.dispatch import crossarray
from crossfit.dataset.base import EmbeddingDatataset
from crossfit.report.beir.embed import embed
from crossfit.calculate.aggregate import Aggregator
from crossfit.metric.continuous.mean import Mean
from crossfit.metric.ranking import NDCG, Precision, Recall, SparseBinaryLabels, SparseRankings


class BeirMetricAggregator(Aggregator):
    def __init__(
        self,
        ks: List[int],
        pre=None,
        post_group=None,
        post=None,
        groupby=None,
        metrics=[NDCG, Precision, Recall],
        # metrics=[Precision],
    ):
        super().__init__(None, pre=pre, post_group=post_group, post=post, groupby=groupby)
        self.ks = ks
        self.metrics = metrics

    def prepare(self, df):
        encoder = self.create_label_encoder(df, ["corpus-index-pred", "corpus-index-obs"])
        obs_csr = self.create_csr_matrix(df["corpus-index-obs"], df["score-obs"], encoder)
        pred_csr = self.create_csr_matrix(df["corpus-index-pred"], df["score-pred"], encoder)

        # TODO: Fix dispatch
        labels = SparseBinaryLabels(CrossSparse.from_matrix(obs_csr))
        rankings = SparseRankings(CrossSparse.from_matrix(pred_csr))

        outputs = {}
        with crossarray:
            for metric in self.metrics:
                for k in self.ks:
                    metric_at_k = metric(k=k)
                    result = metric_at_k.score(labels, rankings)

                    # TODO: Does this make sense?
                    result = np.nan_to_num(result)
                    result = np.where(result > 1, 1, result)

                    outputs[metric_at_k.name()] = Mean.from_array(result, axis=0)

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
            console.rule(f": ".join(self.result_df.index[i]))
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
    model_name: str,
    partition_num: int = 50_000,
    ks=[1, 3, 5, 10],
    overwrite=False,
    out_dir=None,
    client=None,
    groupby=["split"],
    tiny_sample=False,
    dense_search=True,
) -> BeirReport:
    embeddings: EmbeddingDatataset = embed(
        dataset_name,
        model_name=model_name,
        partition_num=partition_num,
        overwrite=overwrite,
        out_dir=out_dir,
        client=client,
        dense_search=dense_search,
        tiny_sample=tiny_sample,
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

    aggregator = BeirMetricAggregator(ks)
    aggregator = Aggregator(aggregator, groupby=groupby, name="")

    results = aggregate(joined, aggregator, to_frame=True)

    return BeirReport(results)
