import gc

import cupy as cp
import cudf
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from tqdm import tqdm

from crossfit.op.base import Op
from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.hf.memory import HFMemoryEstimator
from crossfit.backend.torch.loader import SortedSeqLoader


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

        max_memory = int(self.max_mem.split("GB")[0]) / 2
        self.memory_estimator = HFMemoryEstimator(max_memory, cfg)
        self.model = model.to("cuda")
        self.cfg = cfg

    @torch.no_grad()
    def call(self, data, partition_info=None):
        progress_bar = tqdm(
            total=len(data),
            position=int(self.worker_name),
            desc=f"GPU: {self.worker_name}, Part: {partition_info['number']}",
        )
        predictor = SortedSeqLoader(
            data[["input_ids", "attention_mask"]],
            self.memory_estimator,
            progress_bar=progress_bar,
        ).map(self.model)

        data = predictor.sort_df(data)
        all_embeddings_ls = []
        for output in predictor:
            all_embeddings_ls.append(output["sentence_embedding"])

        out = cudf.DataFrame()
        out.index = data.index
        all_embeddings = torch.vstack(all_embeddings_ls)
        embedding = cp.asarray(all_embeddings)
        out["embedding"] = create_list_series_from_2d_ar(embedding, out.index)

        gc.collect()

        return out

    def meta(self):
        return {"embedding": "float32"}


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
