import gc

import cupy as cp
import cudf
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from cudf.core.subword_tokenizer import _cast_to_appropriate_type
from tqdm import tqdm

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.op.base import Op


class Embedder(Op):
    def __init__(
        self,
        model_name: str,
        pre=None,
        cols=False,
        default_batch_size=1024,
        max_mem: str = "16GB",
    ):
        super().__init__(pre, cols)
        self.model_name = model_name
        self.batch_size = default_batch_size
        self.max_mem = max_mem

    def setup(self):
        model = SentenceTransformer(self.model_name, device="cuda")
        cfg = AutoConfig.from_pretrained("sentence-transformers/" + self.model_name)
        cfg.num_parameters = sum(p.numel() for p in model.parameters())

        self.model = model.to("cuda")
        self.cfg = cfg

    def call(self, data, partition_info=None):
        all_embeddings_ls = []
        sentences = data
        num_rows = len(sentences)
        seq_len = len(sentences["input_ids"][sentences.index[0]])
        tokenized_d = {
            "input_ids": sentences["input_ids"].list.leaves.values.reshape(-1, seq_len),
            "attention_mask": sentences["attention_mask"].list.leaves.values.reshape(-1, seq_len),
        }

        del sentences["input_ids"]
        del sentences["attention_mask"]

        num_tokens = (tokenized_d["input_ids"] != 0).sum(axis=1)
        sentences["num_tokens"] = num_tokens
        sorted_indices = sentences["num_tokens"].argsort()
        sentences = sentences.sort_values(by="num_tokens")
        tokenized_d = {
            key: _cast_to_appropriate_type(val[sorted_indices], "pt")
            for key, val in tokenized_d.items()
        }

        progress = tqdm(
            total=num_rows,
            position=int(self.worker_name),
            desc=f"GPU: {self.worker_name}, Part: {partition_info['number']}",
        )

        free_memory = int(self.max_mem.split("GB")[0])
        free_memory = free_memory / 2  # Add safety margin

        _tokens = sentences["num_tokens"].reset_index(drop=True)
        splits = find_optimal_splits(_tokens, self.batch_size, self.cfg, free_memory)

        current = 0
        with torch.no_grad():
            for split in splits:
                end = min(split, num_rows)
                batch = {key: val[current:end] for key, val in tokenized_d.items()}
                clip_len = min(_tokens[end - 1], self.cfg.max_position_embeddings)
                batch = {key: val[:, :clip_len] for key, val in batch.items()}

                try:
                    model_output = self.model(batch)
                    all_embeddings_ls.append(model_output["sentence_embedding"])
                    progress.update(split - current)
                    current = split
                except IndexError:
                    pass  # Skip this batch

        out = cudf.DataFrame()
        out.index = sentences.index
        all_embeddings = torch.vstack(all_embeddings_ls)
        embedding = cp.asarray(all_embeddings)
        out["embeddings"] = create_list_series_from_2d_ar(embedding, out.index)

        gc.collect()

        return out

    def meta(self):
        return {"embeddings": "float32"}


def estimate_memory(cfg, max_num_tokens, batch_size):
    """
    Estimates the memory consumption for a given SentenceTransformer config and batch.
    Args:
        cfg (transformers.AutoConfig): Configuration object for SentenceTransformer model
        max_num_tokens (int): Maximum number of tokens in a batch
        batch_size (int): Batch size

    Returns:
        total_memory_gb (float): Estimated total memory in gigabytes
    """

    # Model parameters
    total_params = cfg.num_parameters
    model_size = total_params * 4  # in bytes (float32 for each parameter)

    # Hidden size and number of layers
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers

    # Activation memory: assuming a forward and backward pass (even if it's not training, for worst case)
    # The '3' includes memory for input, output and intermediate states for each layer.
    activation_memory = num_layers * batch_size * max_num_tokens * hidden_size * 4 * 3  # in bytes

    # Sum up
    total_memory = model_size + activation_memory  # in bytes
    total_memory_gb = total_memory / (1024**3)  # Convert to GBs

    return total_memory_gb


def find_optimal_splits(
    num_tokens,
    initial_batch_size,
    cfg,
    max_mem_gb,
):
    splits = []
    i = 0
    doubling_factor = 2
    max_doubling_attempts, max_steps = 8, 8
    dynamic_step_size = initial_batch_size
    decreasing_attempts = 0

    while i < len(num_tokens):
        best_fit_e_ind = i + initial_batch_size  # Initialize to at least initial_batch_size

        # Try aggressive doubling first
        for doubling_i in range(max_doubling_attempts):
            tentative_e_ind = i + best_fit_e_ind * doubling_factor  # Double the last best fit
            tentative_e_ind = min(tentative_e_ind, len(num_tokens))
            max_token = int(num_tokens[tentative_e_ind - 1])
            est_memory = estimate_memory(cfg, max_token, int(tentative_e_ind - i))

            if est_memory <= max_mem_gb:
                best_fit_e_ind = tentative_e_ind
            else:
                max_doubling_attempts = doubling_i  # Reduce max doubling attempts
                break  # Exit loop if we exceed memory limit

        for _ in range(max_steps):
            tentative_e_ind = best_fit_e_ind + dynamic_step_size  # Add dynamic step size
            tentative_e_ind = min(tentative_e_ind, len(num_tokens))
            max_token = int(num_tokens[tentative_e_ind - 1])
            est_memory = estimate_memory(cfg, max_token, int(tentative_e_ind - i))

            if est_memory <= max_mem_gb:
                best_fit_e_ind = tentative_e_ind
                break
            else:
                dynamic_step_size //= 2  # halve the step size
                decreasing_attempts += 1

        splits.append(best_fit_e_ind)
        i = best_fit_e_ind  # Move to the next batch

    return splits
