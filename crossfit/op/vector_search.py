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

from typing import overload

import cudf
import cupy as cp
import dask.dataframe as dd
from cuml.dask.neighbors import NearestNeighbors
from dask import delayed
from dask_cudf import from_delayed
from pylibraft.neighbors.brute_force import knn

from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar
from crossfit.backend.dask.cluster import global_dask_client
from crossfit.dataset.base import EmbeddingDatataset
from crossfit.op.base import Op


class VectorSearchOp(Op):
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


class ExactSearchOp(VectorSearchOp):
    def search_tensors(self, queries, corpus):
        raise NotImplementedError()

    def call_part(self, queries, items):
        query_emb = _get_embedding_cupy(queries, self.embedding_col, normalize=self.normalize)
        item_emb = _get_embedding_cupy(items, self.embedding_col, normalize=self.normalize)

        results, indices = self.search_tensors(query_emb, item_emb)

        dfs = []
        for i in range(self.k):
            df = cudf.DataFrame()
            df["query-id"] = queries["_id"]
            df["query-index"] = queries["index"]
            df["corpus-index"] = items["index"].values[indices[:, i]]
            df["score"] = results[:, i]

            dfs.append(df)

        out = cudf.concat(dfs)

        return out

    def call(self, queries, items):
        query_emb = _get_embedding_cupy(queries, self.embedding_col, normalize=self.normalize)
        item_emb = _get_embedding_cupy(items, self.embedding_col, normalize=self.normalize)

        results, indices = self.search_tensors(query_emb, item_emb)

        df = cudf.DataFrame(index=queries.index)
        df["query-id"] = queries["_id"]
        df["query-index"] = queries["index"]
        df["corpus-index"] = create_list_series_from_1d_or_2d_ar(
            items["index"].values[indices], df.index
        )
        df["score"] = create_list_series_from_1d_or_2d_ar(results, df.index)

        return df

    def reduce(self, grouped):
        num_groups = len(grouped)
        scores = grouped["score"].list.leaves.values
        indices = grouped["corpus-index"].list.leaves.values
        indices_flattened = indices.reshape(num_groups, -1)
        scores_flattened = scores.reshape(num_groups, -1)

        desc = float(scores_flattened[0][0]) > float(scores_flattened[0][1])
        scores_ordered = -scores_flattened if desc else scores_flattened
        sorted = cp.argsort(scores_ordered, axis=1)[:, : self.k]
        topk_scores = cp.take_along_axis(scores_flattened, sorted, axis=1)
        topk_indices = cp.take_along_axis(indices_flattened, sorted, axis=1)

        grouped = grouped.reset_index()

        reduced = cudf.DataFrame(index=grouped.index)
        reduced["query-index"] = grouped["query-index"]
        reduced["query-id"] = grouped["query-id"]
        reduced["score"] = create_list_series_from_1d_or_2d_ar(topk_scores, reduced.index)
        reduced["corpus-index"] = create_list_series_from_1d_or_2d_ar(topk_indices, reduced.index)

        reduced = reduced.set_index("query-index", drop=False)

        return reduced

    def call_dask(self, queries, items, partition_num=10_000):
        partitions = max(int(len(items) / partition_num), 1)
        if not partitions % 2 == 0:
            partitions += 1
        _items = items.repartition(npartitions=partitions)

        delayed_cross_products = []
        for i in range(queries.npartitions):
            query_part = queries.get_partition(i)

            for j in range(_items.npartitions):
                item_part = _items.get_partition(j)

                delayed_cross_product = delayed(self.call_part)(
                    query_part,
                    item_part,
                )
                delayed_cross_products.append(delayed_cross_product)

        result_ddf = from_delayed(delayed_cross_products)

        return (
            result_ddf.groupby("query-index").agg(
                {
                    "query-id": "first",
                    "score": list,
                    "corpus-index": list,
                },
            )
        ).map_partitions(
            self.reduce,
            meta={
                "query-index": "int32",
                "query-id": "object",
                "score": "object",
                "corpus-index": "object",
            },
        )


class RaftExactSearch(ExactSearchOp):
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
        self.normalize = normalize

    def search_tensors(self, queries, corpus):
        results, indices = knn(dataset=corpus, queries=queries, k=self.k, metric=self.metric)

        return cp.asarray(results), cp.asarray(indices)


class CuMLVectorSearch(VectorSearchOp):
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

    def fit(self, items, **kwargs):
        knn = NearestNeighbors(
            n_neighbors=self.k,
            algorithm=self.algorithm,
            client=global_dask_client(),
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

        distances, indices = knn.kneighbors(query_ddf_per_dim)

        df = distances
        for i in range(len(indices.columns)):
            df[f"i_{i}"] = indices[i]
        df["index"] = query_ddf_per_dim.index.values

        def join_map(part, n_neighbors: int):
            distances = part.values[:, :n_neighbors].astype("float32")
            # index is last col
            indices = part.values[:, n_neighbors:-1].astype("int32")

            assert indices.shape == distances.shape

            df = cudf.DataFrame()
            df.index = part["index"].values
            df["corpus-index"] = create_list_series_from_1d_or_2d_ar(indices, df.index)
            df["score"] = create_list_series_from_1d_or_2d_ar(distances, df.index)

            return df

        output = df.map_partitions(
            join_map,
            n_neighbors=self.k,
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


class CuMLANNSearch(CuMLVectorSearch):
    def __init__(self, *args, **kwargs):
        algorithm = kwargs.pop("algorithm", None) or "auto"
        super().__init__(algorithm=algorithm, *args, **kwargs)


class CuMLExactSearch(CuMLVectorSearch):
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
    output = output.sort_values("index").drop(labels=["index"], axis=1)

    return output
