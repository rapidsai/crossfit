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

import gc

import cudf
import cupy as cp
import torch

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.loader import DEFAULT_BATCH_SIZE, InMemoryLoader, SortedSeqLoader
from crossfit.backend.torch.model import Model
from crossfit.op.base import Op


class Embedder(Op):
    def __init__(
        self,
        model: Model,
        pre=None,
        cols=False,
        keep_cols=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_mem: str = "16GB",
        sorted_data_loader: bool = True,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model = model
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2.5
        self.sorted_data_loader = sorted_data_loader

    def setup(self):
        self.model.load_on_worker(self)

    @torch.no_grad()
    def call(self, data, partition_info=None):
        index = data.index
        if self.sorted_data_loader:
            loader = SortedSeqLoader(
                data[["input_ids", "attention_mask"]],
                self.model,
                progress_bar=self.create_progress_bar(len(data), partition_info),
                initial_batch_size=self.batch_size,
            )
        else:
            loader = InMemoryLoader(
                data[["input_ids", "attention_mask"]],
                batch_size=self.batch_size,
                progress_bar=self.create_progress_bar(len(data), partition_info),
                max_seq_len=self.model.max_seq_length(),
            )

        all_embeddings_ls = []
        for output in loader.map(self.model.get_model(self)):
            all_embeddings_ls.append(output["sentence_embedding"])

        out = cudf.DataFrame(index=index)
        embedding = cp.asarray(torch.vstack(all_embeddings_ls))
        _index = loader.sort_column(index.values) if self.sorted_data_loader else index
        out["embedding"] = create_list_series_from_2d_ar(embedding, _index)

        gc.collect()
        torch.cuda.empty_cache()

        return out

    def meta(self):
        return {"embedding": "float32"}
