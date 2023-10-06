import functools as ft
from typing import overload

import cudf
import cupy as cp
import dask.dataframe as dd
from cuml.dask.neighbors import NearestNeighbors
from dask import delayed
from dask_cudf import from_delayed
from pylibraft.neighbors.brute_force import knn

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.dataset.base import EmbeddingDatataset
from crossfit.op.base import Op


class DenseSearchOp(Op):
    @overload
    def __call__(self, queries, items, partition_info=None):
        ...

    @overload
    def __call__(self, embedding_data: EmbeddingDatataset, partition_info=None):
        ...

    def __call__(self, data, *args, **kwargs):
        print(f"Doing vector search: {self.__class__.__name__}...")

        if isinstance(data, EmbeddingDatataset):
            queries = data.query.ddf()
            items = data.item.ddf()

            return super().__call__(queries, items, **kwargs)

        return super().__call__(data, *args, **kwargs)


class RaftExactSearch(DenseSearchOp):
    def __init__(
        self,
        k: int,
        pre=None,
        embedding_col="embedding",
        metric="sqeuclidean",
        normalize=True,
        keep_cols=None,
    ):
        super().__init__(pre=pre, keep_cols=keep_cols)
        self.k = k
        self.metric = metric
        self.embedding_col = embedding_col
        self.normalize = normalize

    def call(self, queries, items):
        items = items.reset_index()
        query_emb = _get_embedding_cupy(queries, self.embedding_col, self.normalize)
        item_emb = _get_embedding_cupy(items, self.embedding_col, self.normalize)

        results, indices = knn(dataset=item_emb, queries=query_emb, k=self.k, metric=self.metric)
        results, indices = cp.asarray(results), cp.asarray(indices)

        indices_df = cudf.DataFrame({"key": indices.reshape(-1)})
        ids_df = cudf.DataFrame({"key": items.index, "value": items["index"]})
        merged_df = indices_df.merge(ids_df, on="key", how="left")
        item_ids = merged_df["value"].values.reshape(-1, self.k)

        df = cudf.DataFrame()
        df.index = queries.index
        df["query-id"] = queries["_id"]
        df["query-index"] = queries["index"]
        df["corpus-index"] = create_list_series_from_2d_ar(item_ids, queries["index"])
        df["score"] = create_list_series_from_2d_ar(results, queries["index"])

        return df

    def reduce_dask(self, df):
        df = df.reset_index()
        grouped = (
            df.groupby("query-id")
            .agg(
                {
                    "query-index": "first",
                    "score": list,
                    "corpus-index": list,
                }
            )
            .reset_index()
        )

        num_groups, num_parts = len(grouped), int(grouped["score"].list.len().iloc[0])
        scores = grouped["score"].list.leaves.values.reshape(num_groups, num_parts, -1)
        indices = grouped["corpus-index"].list.leaves.values.reshape(num_groups, num_parts, -1)
        indices_flattened = indices.reshape(num_groups, -1)
        scores_flattened = scores.reshape(num_groups, -1)
        sorted = cp.argsort(-scores_flattened, axis=1)[:, : self.k]
        sorted_scores = cp.take_along_axis(scores_flattened, sorted, axis=1)
        sorted_indices = cp.take_along_axis(indices_flattened, sorted, axis=1)

        out = cudf.DataFrame()
        out.index = grouped.index
        out["query-id"] = grouped["query-id"]
        out["query-index"] = grouped["query-index"]
        out["corpus-index"] = create_list_series_from_2d_ar(sorted_indices, out.index)
        out["score"] = create_list_series_from_2d_ar(sorted_scores, out.index)

        return out

    def call_dask(self, queries, items):
        delayed_cross_products = []
        for i in range(queries.npartitions):
            query_part = queries.get_partition(i)

            for j in range(items.npartitions):
                item_part = items.get_partition(j)

                delayed_cross_product = delayed(self.call)(
                    query_part,
                    item_part,
                )
                delayed_cross_products.append(delayed_cross_product)

        result_ddf = from_delayed(delayed_cross_products)

        output = result_ddf.set_index("query-index")
        output = output.map_partitions(
            self.reduce_dask,
            meta={
                "query-id": "object",
                "query-index": "object",
                "corpus-index": "object",
                "score": "float32",
            },
        )

        return output


class CuMLDenseSearch(DenseSearchOp):
    def __init__(
        self,
        k: int,
        algorithm: str,
        pre=None,
        embedding_col="embedding",
        metric="cosine",
        normalize=True,
        keep_cols=None,
    ):
        super().__init__(pre=pre, keep_cols=keep_cols)
        self.k = k
        self.algorithm = algorithm
        self.metric = metric
        self.embedding_col = embedding_col
        self.metric = metric
        self.normalize = normalize

    def fit(self, items, client=None, **kwargs):
        knn = NearestNeighbors(
            n_neighbors=self.k,
            algorithm=self.algorithm,
            client=client,
            metric=self.metric,
            **kwargs,
        )
        print("Building nearest neighbor index for items...")

        item_ddf = items
        if hasattr(items, "ddf"):
            item_ddf = items.ddf()

        embs = _per_dim_ddf(item_ddf, self.embedding_col, normalize=self.normalize)
        knn.fit(embs)

        return knn

    def query(self, knn, queries):
        query_ddf = queries
        if hasattr(queries, "ddf"):
            query_ddf = queries.ddf()

        query_ddf_per_dim = _per_dim_ddf(query_ddf, self.embedding_col, normalize=self.normalize)
        query_ddf_per_dim = query_ddf_per_dim.sort_values("index").drop(labels=["index"], axis=1)

        distances, indices = knn.kneighbors(query_ddf_per_dim)

        df = distances
        for i in range(len(indices.columns)):
            df[f"i_{i}"] = indices[i]
        df["index"] = query_ddf_per_dim.index.values

        def join_map(part, n_neighbors: int):
            distances = part.values[:, :n_neighbors].astype("float32")
            indices = part.values[:, n_neighbors : 2 * n_neighbors].astype("int32")

            df = cudf.DataFrame()
            df.index = part["index"].values
            df["corpus-index"] = create_list_series_from_2d_ar(indices, df.index)
            df["score"] = create_list_series_from_2d_ar(distances, df.index)

            return df

        output = df.map_partitions(
            ft.partial(join_map, n_neighbors=self.k),
            meta={"corpus-index": "object", "score": "object"},
        )

        output["query-index"] = output.index.values

        output = output.merge(
            query_ddf[["_id"]], left_index=True, right_index=True, how="left"
        ).rename(columns={"_id": "query-id"})

        return output

    def call_dask(self, queries, items):
        knn = self.fit(items)

        return self.query(knn, queries)


class CuMLANNSearch(CuMLDenseSearch):
    def __init__(self, *args, **kwargs):
        algorithm = kwargs.pop("algorithm", None) or "auto"
        super().__init__(algorithm=algorithm, *args, **kwargs)


class CuMLExactSearch(CuMLDenseSearch):
    def __init__(self, *args, **kwargs):
        algorithm = kwargs.pop("algorithm", None) or "brute"
        super().__init__(algorithm=algorithm, *args, **kwargs)


def _get_embedding_cupy(data, embedding_col, normalize=True):
    dim = len(data.head()[embedding_col].iloc[0])

    embs = data[embedding_col].list.leaves.values.reshape(-1, dim).astype("float32")

    if normalize:
        embs = embs / cp.linalg.norm(embs, axis=1, keepdims=True)

    return embs


def _per_dim_ddf(
    data: dd.DataFrame, embedding_col: str, index_col: str = "index", normalize: bool = True
) -> dd.DataFrame:
    dim = len(data.head()[embedding_col].iloc[0])

    def to_map(part, dim):
        values = part[embedding_col].list.leaves.values.reshape(-1, dim).astype("float32")
        if normalize:
            values = values / cp.linalg.norm(values, axis=1, keepdims=True)

        out_part = cudf.DataFrame(values)
        out_part.index = part[index_col].values
        return out_part

    meta = {i: "float32" for i in range(int(dim))}
    output = data.map_partitions(to_map, dim=dim, meta=meta)

    output["index"] = output.index.values
    #output = output.sort_values("index").drop(labels=["index"], axis=1)

    return output
