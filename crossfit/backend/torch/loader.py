# Copyright 2023-2024 NVIDIA CORPORATION
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

import warnings
from itertools import islice
from typing import Dict, overload, Optional

import torch

from crossfit.backend.torch.model import Model
from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.dispatch import CrossFrame

DEFAULT_BATCH_SIZE = 512


class InMemoryLoader:
    @overload
    def __init__(self, data: Dict[str, torch.Tensor], batch_size: int, progress_bar=None):
        ...

    @overload
    def __init__(self, data: CrossFrame, batch_size: int, progress_bar=None):
        ...

    def __init__(self, data, batch_size: int, progress_bar=None, max_seq_len=None, padding_side:str = "right"):
        self.data = CrossFrame(data).cast(torch.Tensor)
        self.tensor_dict = self.data.to_dict()
        self._batch_size = batch_size
        self.num_rows = len(next(iter(self.tensor_dict.values())))
        self.current_idx = 0
        self._to_map = []
        self.progress_bar = progress_bar
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side

    def map(self, fn):
        self._to_map.append(fn)
        return self

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.num_rows:
            raise StopIteration

        batch_size = self.batch_size
        end = batch_size + self.current_idx

        batch = {key: val[self.current_idx : end] for key, val in self.tensor_dict.items()}
        if self.max_seq_len is not None:
            if self.padding_side == "right":
                batch = {key: val[:, : self.max_seq_len] for key, val in batch.items()}
            else:
                batch = {key: val[:, -self.max_seq_len :] for key, val in batch.items()}
        self.current_idx += self.batch_size

        for fn in self._to_map:
            batch = fn(batch)

        if self.progress_bar is not None:
            self.progress_bar.update(batch_size)

        return batch

    def get_batches(self, n):
        return list(islice(self, n))


class SortedSeqLoader(InMemoryLoader):
    @crossarray
    def __init__(
        self,
        data: CrossFrame,
        model: Model,
        sort_key: str = "input_ids",
        initial_batch_size: int = DEFAULT_BATCH_SIZE,
        to_ignore=None,
        progress_bar=None,
    ):
        self.sort_key = sort_key
        self.to_ignore = to_ignore or []
        self.to_ignore.append("seq_length")
        self.model = model
        self.pad_token_id = self.model.load_tokenizer().pad_token_id
        self.padding_side = self.model.load_tokenizer().padding_side

        frame = CrossFrame(data).cast(torch.Tensor)
        seq_length = (frame[sort_key] != self.pad_token_id).sum(axis=1)
        self.sorted_indices = seq_length.argsort(descending=True)
        frame = frame.apply(lambda x: x[self.sorted_indices])
        frame = frame.assign(seq_length=seq_length[self.sorted_indices])

        super().__init__(frame, initial_batch_size, progress_bar=progress_bar)
        self.splits = self._find_optimal_splits()
        # TODO: Debug PRINTS
        print(f"Padding side: {self.padding_side}")
        print(f"Pad token id: {self.pad_token_id}")

    def sort_column(self, col):
        indices = convert_array(self.sorted_indices, type(col))

        return col[indices]

    def sort_df(self, df):
        output = type(df)()
        for col in df.columns:
            output[col] = self.sort_column(df[col])

        return output

    def __next__(self):
        if self.current_idx >= len(self.splits):
            raise StopIteration

        if self.current_idx == 0:
            start = 0
        else:
            start = self.splits[self.current_idx - 1]

        _tokens = self.tensor_dict["seq_length"]

        end = min(self.splits[self.current_idx], self.num_rows)
        while end > start:
            try:
                batch = {
                    key: val[start:end]
                    for key, val in self.tensor_dict.items()
                    if key not in self.to_ignore
                }
                clip_len = min(max(_tokens[start], _tokens[end - 1]), self.model.max_seq_length())
                if self.padding_side == "right":
                    batch = {key: val[:, :clip_len] for key, val in batch.items()}
                else:
                    batch = {key: val[:, -clip_len:] for key, val in batch.items()}

                for fn in self._to_map:
                    batch = fn(batch)

                break
            except torch.cuda.OutOfMemoryError:
                mid = start + (end - start) // 2
                if mid == start:
                    raise
                warnings.warn(
                    f"Not enough memory for a batch size of {end - start}. "
                    f"Retrying with a new batch size of {mid - start}. "
                    f"Consider setting initial batch size to {mid - start}."
                )
                self.splits.insert(self.current_idx, mid)
                end = min(self.splits[self.current_idx], self.num_rows)

        self.current_idx += 1

        if self.progress_bar is not None:
            self.progress_bar.update(end - start)

        return batch

    def _find_optimal_splits(self):
        splits = []
        i = 0
        doubling_factor = 2
        max_doubling_attempts, max_steps = 8, 8
        dynamic_step_size = self.batch_size
        decreasing_attempts = 0

        num_tokens = self.tensor_dict["seq_length"]
        max_seq_length = self.model.max_seq_length()

        while i < len(num_tokens):
            best_fit_e_ind = i + self.batch_size  # Initialize to at least initial_batch_size

            # Try aggressive doubling first
            for doubling_i in range(max_doubling_attempts):
                tentative_e_ind = i + best_fit_e_ind * doubling_factor  # Double the last best fit
                tentative_e_ind = min(tentative_e_ind, len(num_tokens))
                max_token = int(num_tokens[tentative_e_ind - 1])
                est_memory = self.model.estimate_memory(max_token, int(tentative_e_ind - i))

                if est_memory <= self.model.max_mem_gb:
                    best_fit_e_ind = tentative_e_ind
                else:
                    max_doubling_attempts = doubling_i  # Reduce max doubling attempts
                    break  # Exit loop if we exceed memory limit

            for _ in range(max_steps):
                tentative_e_ind = best_fit_e_ind + dynamic_step_size  # Add dynamic step size
                tentative_e_ind = min(tentative_e_ind, len(num_tokens))
                max_token = int(num_tokens[tentative_e_ind - 1])

                est_memory = self.model.estimate_memory(max_token, int(tentative_e_ind - i))
                # The closer we are to the end, the more we penalize the batch size
                penalty_factor = 1 + 5.0 * ((max_token / max_seq_length) ** 2)
                est_memory *= penalty_factor

                if est_memory <= self.model.max_mem_gb:
                    best_fit_e_ind = tentative_e_ind
                    break
                else:
                    dynamic_step_size //= 2  # halve the step size
                    decreasing_attempts += 1

            splits.append(best_fit_e_ind)
            i = best_fit_e_ind  # Move to the next batch

        return splits
