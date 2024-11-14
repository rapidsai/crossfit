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
from functools import lru_cache

import joblib
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from crossfit.backend.torch.hf.memory_curve_utils import fit_memory_estimate_curve
from crossfit.backend.torch.model import Model
from crossfit.dataset.home import CF_HOME
from crossfit.utils.torch_utils import cleanup_torch_cache


class HFModel(Model):
    def __init__(
        self,
        path_or_name: str,
        max_mem_gb: int = 16,
        model_output_type: str = "numeric",
        training: bool = False,
        start_batch_size: int = 1,
        end_batch_size: int = 2048,
        batch_size_increment: int = 256,
        start_seq_len: int = 1,
        seq_len_increment: int = 64,
    ):
        super().__init__(path_or_name, max_mem_gb, model_output_type)
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.batch_size_increment = batch_size_increment
        self.start_seq_len = start_seq_len
        self.seq_len_increment = seq_len_increment
        self._cfg_id = f"cfg_{id(self)}"

        cache_dir = os.path.join(CF_HOME, "memory", self.load_cfg()._name_or_path)
        os.makedirs(cache_dir, exist_ok=True)
        mem_model_path = os.path.join(cache_dir, "mem_model.pkl")
        if os.path.exists(mem_model_path):
            self.mem = joblib.load(mem_model_path)
        else:
            model = self.load_model("cuda") if not training else None
            end_seq_len = self.max_seq_length()
            if not training:
                with torch.no_grad():
                    self.mem = fit_memory_estimate_curve(
                        model=model,
                        path_or_name=self.path_or_name,
                        start_batch_size=start_batch_size,
                        end_batch_size=end_batch_size,
                        start_seq_len=start_seq_len,
                        end_seq_len=end_seq_len,
                        batch_size_increment=batch_size_increment,
                        seq_len_increment=seq_len_increment,
                        mem_model_path=mem_model_path,
                    )
            else:
                self.mem = fit_memory_estimate_curve(
                    model=model,
                    path_or_name=self.path_or_name,
                    start_batch_size=start_batch_size,
                    end_batch_size=end_batch_size,
                    start_seq_len=start_seq_len,
                    end_seq_len=end_seq_len,
                    batch_size_increment=batch_size_increment,
                    seq_len_increment=seq_len_increment,
                    mem_model_path=mem_model_path,
                )

    def load_on_worker(self, worker, device="cuda"):
        setattr(worker, self._model_id, self.load_model(device))
        setattr(worker, self._cfg_id, self.load_cfg())

    def unload_from_worker(self, worker):
        if hasattr(worker, self._model_id):
            delattr(worker, self._model_id)
        if hasattr(worker, self._cfg_id):
            delattr(worker, self._cfg_id)
        cleanup_torch_cache()

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
            max_seq_length = self.load_cfg().max_position_embeddings
        return max_seq_length


class SentenceTransformerModel(HFModel):
    def load_model(self, device="cuda"):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.path_or_name, device="cuda").to(device)

    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name)
