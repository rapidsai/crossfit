from functools import lru_cache
import gc
import logging
import os
from crossfit.dataset.home import CF_HOME
import joblib

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.linear_model import LinearRegression
from crossfit.backend.torch.model import Model

logger = logging.getLogger(__name__)


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
        for batch_size in tqdm(range(1, 2048, 256)):
            for seq_len in list(range(16, max_seq, 64)) + [max_seq]:
                torch.cuda.reset_peak_memory_stats()

                batch = {
                    "input_ids": torch.randint(1, 501, (batch_size, seq_len)).to(device=device),
                    "attention_mask": torch.ones((batch_size, seq_len)).to(device=device),
                }

                try:
                    outputs = model(batch)
                    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
                    X.append([batch_size, seq_len, seq_len**2])
                    y.append(memory_used)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    else:
                        raise e

        self.mem = LinearRegression().fit(np.array(X), np.array(y))
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(self.mem, mem_model_path)

        del outputs
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
        return self.load_cfg().max_position_embeddings


class SentenceTransformerModel(HFModel):
    def load_model(self, device="cuda"):
        model = AutoModel.from_pretrained(self.path_or_name)
        if device == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer

                model = BetterTransformer.transform(model.to(torch.float16)).to(device)
            except ImportError:
                logging.warning(
                    "Loading embedding model without BetterTransformer. "
                    "Install the 'optimum' to make embedding inference faster.  "
                )
        return model

    def get_sentence_embedding(self, inputs, outputs):
        embeddings = self.average_pool(
            outputs["last_hidden_state"], inputs["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def load_cfg(self):
        return AutoConfig.from_pretrained("sentence-transformers/" + self.path_or_name)
