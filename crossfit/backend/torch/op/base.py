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
from typing import Optional

import cudf
import cupy as cp
import torch

from crossfit.backend.cudf.series import (
    create_list_series_from_1d_or_2d_ar,
    create_nested_list_series_from_3d_ar,
)
from crossfit.backend.torch.loader import DEFAULT_BATCH_SIZE, InMemoryLoader, SortedSeqLoader
from crossfit.backend.torch.model import Model
from crossfit.op.base import Op


class Predictor(Op):
    def __init__(
        self,
        model: Model,
        pre=None,
        post=None,
        cols=False,
        keep_cols=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_mem: str = "16GB",
        sorted_data_loader: bool = True,
        model_output_col: Optional[str] = None,
        pred_output_col: str = "preds",
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model = model
        self.post = post
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2.5
        self.sorted_data_loader = sorted_data_loader
        self.model_output_col = model_output_col
        self.pred_output_col = pred_output_col

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
                padding_side=self.model.load_tokenizer().padding_side,
                progress_bar=self.create_progress_bar(len(data), partition_info),
                max_seq_len=self.model.max_seq_length(),
            )

        all_outputs_ls = []
        for output in loader.map(self.model.get_model(self.get_worker())):
            if isinstance(output, dict):
                if self.model_output_col not in output:
                    raise ValueError(f"Column '{self.model_outupt_col}' not found in model output.")
                output = output[self.model_output_col]

            if self.post is not None:
                output = self.post(output)

            all_outputs_ls.append(output)

        out = cudf.DataFrame(index=index)
        outputs = cp.asarray(torch.cat(all_outputs_ls, dim=0))
        _index = loader.sort_column(index.values) if self.sorted_data_loader else index
        if len(outputs.shape) <= 2:
            out[self.pred_output_col] = create_list_series_from_1d_or_2d_ar(outputs, _index)
        elif len(outputs.shape) == 3:
            out[self.pred_output_col] = create_nested_list_series_from_3d_ar(outputs, _index)
        else:
            raise RuntimeError(f"Unexpected output shape: {output.shape}")

        gc.collect()
        torch.cuda.empty_cache()

        return out

    def meta(self):
        return {self.pred_output_col: "float32"}
