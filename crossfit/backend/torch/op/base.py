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

from typing import Optional

import torch

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
        model_output_cols: Optional[list] = None,
        pred_output_col: Optional[str] = None,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model = model
        self.post = post
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2.5
        self.sorted_data_loader = sorted_data_loader

        if model_output_col and model_output_cols:
            raise ValueError("Specify either model_output_col or model_output_cols, not both.")
        elif model_output_col:
            self.model_output_cols = [model_output_col]
        elif model_output_cols:
            self.model_output_cols = model_output_cols
        else:
            self.model_output_cols = None

        if pred_output_col and len(self.model_output_cols) > 1:
            raise ValueError(
                "pred_output_col can only be specified when model_output_cols has a single column."
            )
        self.pred_output_col = pred_output_col or "preds"

    @torch.no_grad()
    def call(self, data, partition_info=None):
        index = data.index.copy()
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
        del data
        all_outputs_ls = []
        for output in loader.map(self.model.get_model(self.get_worker())):
            if isinstance(output, dict):
                if self.model_output_cols:
                    output = {col: output[col] for col in self.model_output_cols if col in output}
                if len(output) == 1:
                    output = list(output.values())[0]
                elif len(output) > 1 and self.model_output_cols is None:
                    raise ValueError(
                        "Model returned more than one output column, but model_output_cols ",
                        "was not specified. Please specify model_output_cols",
                        "to get all model outputs.",
                    )
            if self.post is not None:
                output = self.post(output)
            all_outputs_ls.append(output)
        out = self.model.get_model_output(all_outputs_ls, index, loader, self.pred_output_col)
        return out

    def meta(self):
        dtype = "object" if self.model.model_output_type == "string" else "float32"
        if self.model_output_cols and len(self.model_output_cols) > 1:
            return {col: dtype for col in self.model_output_cols}
        else:
            return {self.pred_output_col: dtype}
