import os
import shutil
import functools as ft

from cuml.dask.neighbors import NearestNeighbors
import cudf

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.dataset.home import CF_HOME
from crossfit.dataset.base import IRDataset, EmbeddingDatataset, Dataset
from crossfit.dataset.load import load_dataset
from crossfit import op


def embed(
    dataset_name: str,
    model_name: str,
    partition_num: int = 50_000,
    overwrite=False,
    out_dir=None,
    dense_search=True,
    client=None,
) -> EmbeddingDatataset:
    dataset: IRDataset = load_dataset("beir/" + dataset_name, overwrite=overwrite)

    out_dir = out_dir or CF_HOME
    emb_dir = os.path.join(out_dir, "processed", "beir", dataset_name, "emb", model_name)

    if os.path.exists(emb_dir):
        if overwrite:
            print("Embedding directory {} already exists. Overwriting.".format(emb_dir))
            shutil.rmtree(emb_dir)

        else:
            return EmbeddingDatataset.from_dir(emb_dir, data=dataset)

    dfs = []
    for dtype in ["query", "item"]:
        df = getattr(dataset, dtype).ddf()
        partitions = max(int(len(df) / partition_num), 1)
        if not partitions % 2 == 0:
            partitions += 1

        print(f"Embedding {dataset_name} {dtype} ({partitions} parts)...")

        pipe = op.Sequential(
            op.Tokenizer(model_name, cols=["text"]),
            op.Embedder(model_name),
            repartition=partitions,
            keep_cols=["index", "_id"],
        )
        embeddings = pipe(df)

        embeddings.to_parquet(os.path.join(emb_dir, dtype))
        dfs.append(df)

    output: EmbeddingDatataset = EmbeddingDatataset.from_dir(emb_dir, data=dataset)
    if dense_search:
        topk_df = query_kneighbors(output, client=client)
        pred_path = os.path.join(emb_dir, "predictions")
        topk_df.to_parquet(pred_path)
        output.predictions = Dataset(pred_path)

    return output


def item_knn(items, n_neighbors=50, client=None, **kwargs) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=n_neighbors, client=client, **kwargs)
    print("Building ANN-index for items...")

    item_ddf = items.ddf()
    item_ddf_per_dim = per_dim_ddf(item_ddf.set_index("index"))
    knn.fit(item_ddf_per_dim)

    return knn


def query_kneighbors(embeddings, knn_index=None, n_neighbors=50, client=None):
    knn_index = knn_index or item_knn(embeddings.item, n_neighbors=n_neighbors, client=client)

    print("Querying ANN-index for queries...")
    query_ddf = embeddings.query.ddf()
    _query_ddf = query_ddf.set_index("index")
    query_ddf_per_dim = per_dim_ddf(_query_ddf)

    distances, indices = knn_index.kneighbors(query_ddf_per_dim)

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
        ft.partial(join_map, n_neighbors=n_neighbors),
        meta={"corpus-index": "object", "score": "float32"},
    )

    output["query-id"] = _query_ddf["_id"]
    output["query-index"] = _query_ddf.index

    return output


def per_dim_ddf(data):
    dim = len(data.head()["embedding"].iloc[0])

    def to_map(part, dim):
        df = cudf.DataFrame(part["embedding"].list.leaves.values.reshape(-1, dim).astype("float32"))

        return df

    meta = {i: "float32" for i in range(int(dim))}
    output = data.map_partitions(to_map, dim=dim, meta=meta)

    return output
