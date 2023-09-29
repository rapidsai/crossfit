import os
import shutil
import functools as ft

from cuml.dask.neighbors import NearestNeighbors
import cudf

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.dataset.home import CF_HOME
from crossfit.dataset.base import IRDataset, EmbeddingDatataset, IRData, Dataset
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
        print(f"Embedding {dataset_name} {dtype}...")
        df = getattr(dataset, dtype).ddf()

        partitions = max(int(len(df) / partition_num), 1)
        if not partitions % 2 == 0:
            partitions += 1

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
    knn.fit(items.per_dim_ddf())

    return knn


def query_kneighbors(embeddings, knn_index=None, n_neighbors=50, client=None):
    knn_index = knn_index or item_knn(embeddings.item, n_neighbors=n_neighbors, client=client)

    print("Querying ANN-index for queries...")
    query_ddf = embeddings.query.ddf()
    distances, indices = knn_index.kneighbors(embeddings.query.per_dim_ddf(query_ddf))

    index = query_ddf["index"]
    distances.index = index
    indices.index = index

    def join_map(part, num_cols):
        distances = part.values[:, :num_cols].astype("float32")
        indices = part.values[:, num_cols:].astype("int32")

        df = cudf.DataFrame()
        df.index = part.index
        df["corpus-index"] = create_list_series_from_2d_ar(indices, df.index)
        df["score"] = create_list_series_from_2d_ar(distances, df.index)

        return df

    joined = distances.join(indices, lsuffix="d", rsuffix="i")
    joined = joined.map_partitions(
        ft.partial(join_map, num_cols=len(distances.columns)),
        meta={"corpus-index": "object", "score": "float32"},
    )

    joined["query-id"] = query_ddf["_id"]
    joined["query-index"] = index

    return joined


def evaluate(
    dataset_name: str,
    model_name: str,
    partition_num: int = 50_000,
    split="test",
    overwrite=False,
    out_dir=None,
    client=None,
):
    embeddings = embed(
        dataset_name,
        model_name=model_name,
        partition_num=partition_num,
        overwrite=overwrite,
        out_dir=out_dir,
        client=client,
    )

    data: IRData = getattr(embeddings.data, split)
    joined = data.join_predictions(embeddings.predictions)
