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
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from crossfit.backend.torch.hf.memory_curve_utils import fit_memory_estimate_curve
from crossfit.backend.torch.model import Model


class HFModel(Model):
    def __init__(
        self,
        path_or_name: str,
        max_mem_gb: int = 16,
        training: bool = False,
        start_batch_size: int = 1,
        end_batch_size: int = 2048,
        batch_size_increment: int = 256,
        start_seq_len: int = 1,
        seq_len_increment: int = 64,
    ):
        super().__init__(path_or_name, max_mem_gb)
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.batch_size_increment = batch_size_increment
        self.start_seq_len = start_seq_len
        self.seq_len_increment = seq_len_increment

        model = self.load_model("cuda") if not training else None
        if not training:
            with torch.no_grad():
                self.mem = fit_memory_estimate_curve(
                    model,
                    self.path_or_name,
                    start_batch_size,
                    end_batch_size,
                    batch_size_increment,
                    start_seq_len,
                    seq_len_increment,
                )
        else:
            self.mem = fit_memory_estimate_curve(
                model,
                self.path_or_name,
                start_batch_size,
                end_batch_size,
                batch_size_increment,
                start_seq_len,
                seq_len_increment,
            )

    def load_on_worker(self, worker, device="cuda"):
        worker.torch_model = self.load_model(device)
        worker.cfg = self.load_cfg()

    def unload_from_worker(self, worker):
        if hasattr(worker, "torch_model"):
            delattr(worker, "torch_model")
        if hasattr(worker, "cfg"):
            delattr(worker, "cfg")
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self, device="cuda"):
        return AutoModel.from_pretrained(self.path_or_name).to(device)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.path_or_name)

    @lru_cache(maxsize=1)
    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name)

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        predicted_memory = self.mem.predict(
            np.array([[batch_size, max_num_tokens, max_num_tokens**2]])
        )
        return predicted_memory[0] / 1024  # Convert from MB to GB

    def max_seq_length(self) -> int:
        max_seq_length = self.load_tokenizer().model_max_length
        # Guard against the HF bug
        # which sets max_seq_length to max(int) for some models
        if max_seq_length > 1e5:
            max_seq_length = AutoConfig.from_pretrained(self.path_or_name).max_position_embeddings
        return max_seq_length


class SentenceTransformerModel(HFModel):
    def load_model(self, device="cuda"):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.path_or_name, device="cuda").to(device)

    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name)
