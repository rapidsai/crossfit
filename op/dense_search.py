from typing import overload
import functools as ft

import cupy as cp
import cudf
from dask import delayed
import dask_cudf
from dask_cudf import from_delayed
from cuml.dask.neighbors import NearestNeighbors
from pylibraft.neighbors.brute_force import knn

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.op.base import Op
from crossfit.dataset.base import EmbeddingDatataset


class DenseSearchOp(Op):
    @overload
    def __call__(self, queries, items, partition_info=None):
        ...

    @overload
    def __call__(self, embedding_data: EmbeddingDatataset, partition_info=None):
        ...

    def __call__(self, data, *args, **kwargs):
        if isinstance(data, EmbeddingDatataset):
            queries = data.query.ddf()
            items = data.item.ddf()

            return super().__call__(queries, items, **kwargs)

        return super().__call__(data, *args, **kwargs)


class ExactSearch(DenseSearchOp):
    def __init__(
        self,
        k: int,
        pre=None,
        embedding_col="embedding",
        metric="cosine",
        normalize=True,
        keep_cols=None,
    ):
        super().__init__(pre=pre, keep_cols=keep_cols)
        self.k = k
        self.metric = metric
        self.embedding_col = embedding_col

    def call(self, queries, items):
        items = items.reset_index()
        query_emb = _get_embedding_cupy(queries, self.embedding_col)
        item_emb = _get_embedding_cupy(items, self.embedding_col)

        results, indices = knn(dataset=item_emb, queries=query_emb, k=self.k, metric=self.metric)
        results, indices = cp.asarray(results), cp.asarray(indices)

        # Create a DataFrame for 'indices' with a column name 'key'
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

        # Create a new Dask DataFrame from the delayed objects
        result_ddf = from_delayed(delayed_cross_products)
        results = (
            result_ddf.groupby("query-id")
            .apply(
                lambda x: x.nlargest(self.k, "score"),
                meta={
                    "query-id": "object",
                    "query-index": "object",
                    "corpus-index": "object",
                    "score": "float32",
                },
            )
            .reset_index(drop=True)
            .groupby("query-id")
            .agg({"query-index": "first", "corpus-index": list, "score": list})
            .reset_index()
        )

        return results


class CuMLANNSearch(DenseSearchOp):
    def __init__(
        self,
        k: int,
        pre=None,
        embedding_col="embedding",
        metric="cosine",
        normalize=True,
        keep_cols=None,
    ):
        super().__init__(pre=pre, keep_cols=keep_cols)
        self.k = k
        self.metric = metric
        self.embedding_col = embedding_col
        self.metric = metric
        self.normalize = normalize

    def fit(self, items, client=None, **kwargs):
        knn = NearestNeighbors(n_neighbors=self.k, client=client, metric=self.metric, **kwargs)
        print("Building ANN-index for items...")

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

        _query_ddf = query_ddf.set_index("index")
        query_ddf_per_dim = _per_dim_ddf(_query_ddf, self.embedding_col, normalize=self.normalize)

        distances, indices = knn.kneighbors(query_ddf_per_dim)

        distances.index = _query_ddf.index
        indices.index = _query_ddf.index

        df = distances
        for i in range(len(indices.columns)):
            df[f"i_{i}"] = indices[i]

        def join_map(part, n_neighbors: int):
            distances = part.values[:, :n_neighbors].astype("float32")
            indices = part.values[:, n_neighbors:].astype("int32")

            df = cudf.DataFrame()
            df.index = part.index
            df["corpus-index"] = create_list_series_from_2d_ar(indices, df.index)
            df["score"] = create_list_series_from_2d_ar(distances, df.index)

            return df

        output = df.map_partitions(
            ft.partial(join_map, n_neighbors=self.k),
            meta={"corpus-index": "object", "score": "object"},
        )

        output["query-id"] = _query_ddf["_id"]
        output["query-index"] = _query_ddf.index

        return output

    def call_dask(self, queries, items):
        knn = self.fit(items)

        return self.query(knn, queries)


def _get_embedding_cupy(data, embedding_col):
    dim = len(data.head()[embedding_col].iloc[0])

    return data[embedding_col].list.leaves.values.reshape(-1, dim).astype("float32")


def _per_dim_ddf(
    data: dask_cudf.DataFrame, embedding_col: str, normalize: bool = True
) -> dask_cudf.DataFrame:
    dim = len(data.head()[embedding_col].iloc[0])

    def to_map(part, dim):
        values = part[embedding_col].list.leaves.values.reshape(-1, dim).astype("float32")
        if normalize:
            values = values / cp.linalg.norm(values, axis=1, keepdims=True)

        return cudf.DataFrame(values)

    meta = {i: "float32" for i in range(int(dim))}
    output = data.map_partitions(to_map, dim=dim, meta=meta)

    output.index = data.index
    output = output.reset_index().set_index("index", sort=True, drop=True)

    return output
