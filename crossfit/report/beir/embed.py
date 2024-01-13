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

import os
import shutil
from typing import Optional

from crossfit import op
from crossfit.backend.torch.loader import DEFAULT_BATCH_SIZE
from crossfit.backend.torch.model import Model
from crossfit.dataset.base import Dataset, EmbeddingDatataset, IRDataset
from crossfit.dataset.home import CF_HOME
from crossfit.dataset.load import load_dataset
from crossfit.op.vector_search import VectorSearchOp


def embed(
    dataset_name: str,
    model: Model,
    vector_search: Optional[VectorSearchOp] = None,
    partition_num: int = 50_000,
    overwrite: bool = False,
    out_dir: Optional[str] = None,
    tiny_sample: bool = False,
    sorted_data_loader: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingDatataset:
    dataset: IRDataset = load_dataset(
        "beir/" + dataset_name, overwrite=overwrite, tiny_sample=tiny_sample
    )

    out_dir = out_dir or CF_HOME
    processed_name = "processed-test" if tiny_sample else "processed"
    emb_dir = os.path.join(out_dir, processed_name, "beir", dataset_name, "emb", model.path_or_name)

    if os.path.exists(emb_dir):
        if overwrite:
            print("Embedding directory {} already exists. Overwriting.".format(emb_dir))
            shutil.rmtree(emb_dir)

        else:
            return EmbeddingDatataset.from_dir(emb_dir, data=dataset)

    for dtype in ["query", "item"]:
        if os.path.exists(os.path.join(emb_dir, dtype)):
            continue

        df = getattr(dataset, dtype).ddf()
        if partition_num:
            partitions = max(int(len(df) / partition_num), 1)
            if not partitions % 2 == 0:
                partitions += 1
        else:
            partitions = None

        print(f"Embedding {dataset_name} {dtype} ({partitions} parts)...")

        pipe = op.Sequential(
            op.Tokenizer(model, cols=["text"]),
            op.Embedder(model, sorted_data_loader=sorted_data_loader, batch_size=batch_size),
            repartition=partitions,
            keep_cols=["index", "_id"],
        )
        embeddings = pipe(df)

        embeddings.to_parquet(os.path.join(emb_dir, dtype))

    output: EmbeddingDatataset = EmbeddingDatataset.from_dir(emb_dir, data=dataset)
    pred_path = os.path.join(emb_dir, "predictions")

    if not vector_search:
        return output

    topk_df = vector_search(output)
    topk_df.to_parquet(pred_path)
    output.predictions = Dataset(pred_path)

    return output
