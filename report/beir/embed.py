import functools as ft
import os
import shutil
from typing import Optional, Union

import cupy as cp
import cudf
import dask.dataframe as dd
from cuml.dask.neighbors import NearestNeighbors

from crossfit import op
from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.dataset.base import Dataset, EmbeddingDatataset, IRDataset
from crossfit.dataset.home import CF_HOME
from crossfit.dataset.load import load_dataset
from crossfit.op.dense_search import DenseSearchOp, CuMLANNSearch


def embed(
    dataset_name: str,
    model_name: str,
    partition_num: int = 50_000,
    overwrite: bool = False,
    out_dir: Optional[str] = None,
    dense_search: Union[bool, DenseSearchOp] = False,
    n_neighbors: int = 100,
    normalize: bool = True,
    client=None,
    tiny_sample: bool = False,
) -> EmbeddingDatataset:
    dataset: IRDataset = load_dataset(
        "beir/" + dataset_name, overwrite=overwrite, tiny_sample=tiny_sample
    )

    out_dir = out_dir or CF_HOME
    processed_name = "processed-test" if tiny_sample else "processed"
    emb_dir = os.path.join(out_dir, processed_name, "beir", dataset_name, "emb", model_name)

    if os.path.exists(emb_dir):
        if overwrite:
            print("Embedding directory {} already exists. Overwriting.".format(emb_dir))
            shutil.rmtree(emb_dir)

        else:
            return EmbeddingDatataset.from_dir(emb_dir, data=dataset)

    dfs = []
    for dtype in ["query", "item"]:
        if os.path.exists(os.path.join(emb_dir, dtype)):
            continue

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
    pred_path = os.path.join(emb_dir, "predictions")

    if dense_search is False:
        return output

    if dense_search is True:
        dense_search = CuMLANNSearch(n_neighbors, normalize=normalize)

    topk_df = dense_search(output)
    topk_df.to_parquet(pred_path)
    output.predictions = Dataset(pred_path)

    return output
