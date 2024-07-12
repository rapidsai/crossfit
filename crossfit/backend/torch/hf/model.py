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
import os
from functools import lru_cache

import joblib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from crossfit.backend.torch.model import Model
from crossfit.dataset.home import CF_HOME


class HFModel(Model):
    def __init__(self, path_or_name: str, max_mem_gb: int = 16, training=False):
        super().__init__(path_or_name, max_mem_gb)
    
        if not training:
            with torch.no_grad():
                self.fit_memory_estimate_curve()
        else:
            self.fit_memory_estimate_curve()

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

    def fit_memory_estimate_curve(self, model=None):
        remove_model = False
        if model is None:
            remove_model = True
            model = self.load_model(device="cuda")

        cache_dir = os.path.join(CF_HOME, "memory", self.load_cfg()._name_or_path)
        mem_model_path = os.path.join(cache_dir, "mem_model.pkl")

        if os.path.exists(mem_model_path):
            self.mem = joblib.load(mem_model_path)

            return self

        print(f"Fitting memory estimate curve for model: {self.path_or_name}")

        device = next(model.parameters()).device
        X = []
        y = []

        max_seq = self.max_seq_length()
        for batch_size in tqdm(range(2048, 0, -256)):
            if batch_size <= 0:
                continue

            for seq_len in range(max_seq, 0, -64):
                if seq_len <= 0:
                    continue

                torch.cuda.reset_peak_memory_stats()

                batch = {
                    "input_ids": torch.randint(1, 501, (batch_size, seq_len)).to(device=device),
                    "attention_mask": torch.ones((batch_size, seq_len)).to(device=device),
                }

                try:
                    _ = model(batch)
                    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
                    X.append([batch_size, seq_len, seq_len**2])
                    y.append(memory_used)

                except RuntimeError as e:
                    if "out of memory" in str(e) or "out_of_memory" in str(e):
                        pass
                    else:
                        raise e
                finally:
                    del batch
                    if "outputs" in vars():
                        del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

        self.mem = LinearRegression().fit(np.array(X), np.array(y))
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(self.mem, mem_model_path)

        if remove_model:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        predicted_memory = self.mem.predict(
            np.array([[batch_size, max_num_tokens, max_num_tokens**2]])
        )
        return predicted_memory[0] / 1024  # Convert from MB to GB

    def max_seq_length(self) -> int:
        max_seq_length = self.load_tokenizer().model_max_length
        # Gaurd against the HF bug
        # which sets max_seq_length to max(int) for some models
        if max_seq_length > 1e5:
            max_seq_length = AutoConfig.from_pretrained(
                self.model_name_or_path
            ).max_position_embeddings
        return max_seq_length


class SentenceTransformerModel(HFModel):
    def load_model(self, device="cuda"):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.path_or_name, device="cuda").to(device)

    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name)
