# Copyright 2024 NVIDIA CORPORATION
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


import joblib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import PreTrainedModel

from crossfit.utils.model_adapter import adapt_model_input
from crossfit.utils.torch_utils import (
    cleanup_torch_cache,
    get_peak_memory_used,
    reset_memory_tracking,
)


def fit_memory_estimate_curve(
    model: PreTrainedModel,
    path_or_name: str,
    start_batch_size: int = 1,
    end_batch_size: int = 2048,
    batch_size_increment: int = 256,
    start_seq_len: int = 1,
    end_seq_len: int = 2048,
    seq_len_increment: int = 64,
    mem_model_path: str = None,
) -> LinearRegression:
    print(f"Fitting memory estimate curve for model: {path_or_name}")

    device = "cuda"
    X: list[list[int]] = []
    y: list[float] = []

    batch_size_pbar = tqdm(
        range(start_batch_size, end_batch_size + 1, batch_size_increment), desc="Batch size"
    )
    for batch_size in batch_size_pbar:
        seq_len_pbar = tqdm(
            range(start_seq_len, end_seq_len + 1, seq_len_increment),
            desc="Sequence length",
            leave=False,
        )
        for seq_len in seq_len_pbar:
            reset_memory_tracking()
            batch = {
                "input_ids": torch.randint(1, 501, (batch_size, seq_len)).to(device=device),
                "attention_mask": torch.ones((batch_size, seq_len)).to(device=device),
            }

            try:
                _ = adapt_model_input(model, batch)
                memory_used = get_peak_memory_used()
                memory_used = memory_used / (1024**2)  # Convert to MB
                X.append([batch_size, seq_len, seq_len**2])
                y.append(memory_used)

            except RuntimeError as e:
                # Catching run time error because:
                # https://github.com/pytorch/pytorch/issues/133280
                if "out of memory" in str(e) or "out_of_memory" in str(e):
                    # Early stopping for this batch size
                    seq_len_pbar.close()
                    break
                else:
                    raise e
            finally:
                del batch
                if "outputs" in vars():
                    del outputs
                cleanup_torch_cache()

        # Check if we've hit the memory limit for all sequence lengths
        if seq_len == start_seq_len:
            batch_size_pbar.close()
            break

    mem_model = LinearRegression().fit(np.array(X), np.array(y))
    joblib.dump(mem_model, mem_model_path)

    return mem_model
