import os
import shutil

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
            op.Tokenizer(model_name, cols=["text"]), op.Embedder(model_name), repartition=partitions
        )
        df = df.set_index("_id")
        pipe(df).to_parquet(os.path.join(emb_dir, dtype), write_index=True)
        dfs.append(df)

    output = EmbeddingDatataset.from_dir(emb_dir, data=dataset)
    if dense_search:
        topk_df = output.query_kneighbors(client=client)
        pred_path = os.path.join(emb_dir, "predictions")
        topk_df.to_parquet(pred_path, write_index=True)
        output.predictions = Dataset(pred_path)

    return output


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
