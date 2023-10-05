from typing import Dict, overload
from itertools import islice
from crossfit.data.array.dispatch import crossarray

import torch

from crossfit.data.dataframe.dispatch import CrossFrame


class InMemoryLoader:
    @overload
    def __init__(self, data: Dict[str, torch.Tensor], batch_size: int):
        ...

    @overload
    def __init__(self, data: CrossFrame, batch_size: int):
        ...

    def __init__(self, data, batch_size: int):
        self.tensor_dict = CrossFrame(data).cast(torch.Tensor).to_dict()
        self._batch_size = batch_size
        self.num_rows = len(next(iter(self.tensor_dict.values())))
        self.current_idx = 0
        self._to_map = []

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

        next_batch = {key: val[self.current_idx : end] for key, val in self.tensor_dict.items()}

        self.current_idx += self.batch_size

        for fn in self._to_map:
            next_batch = fn(next_batch)

        return next_batch

    def get_batches(self, n):
        return list(islice(self, n))


class MemoryEstimator:
    def __init__(self, max_mem_gb: int):
        self.max_mem_gb = max_mem_gb

    def max_seq_length(self) -> int:
        raise NotImplementedError()

    def estimate(self, max_num_tokens: int, batch_size: int) -> int:
        raise NotImplementedError()

    def __call__(self, max_num_tokens: int, batch_size: int) -> int:
        return self.estimate(max_num_tokens, batch_size)


class SortedSeqLoader(InMemoryLoader):
    @crossarray
    def __init__(
        self,
        data: CrossFrame,
        memory_estimator: MemoryEstimator,
        sort_key: str = "input_ids",
        initial_batch_size: int = 1024,
        to_ignore=None,
        progress_bar=None,
    ):
        self._original_data = data
        self.sort_key = sort_key
        self.to_ignore = to_ignore or []
        self.to_ignore.append("seq_length")
        self.progress_bar = progress_bar
        self.memory_estimator = memory_estimator

        frame = CrossFrame(data).cast(torch.Tensor)
        seq_length = (frame[sort_key] != 0).sum(axis=1)
        self.sorted_indices = seq_length.argsort()
        frame = frame.apply(lambda x: x[self.sorted_indices])
        frame = frame.assign(seq_length=seq_length[self.sorted_indices])

        super().__init__(frame, initial_batch_size)
        self.splits = self.find_optimal_splits()

    def __next__(self):
        if self.current_idx >= self.num_rows:
            raise StopIteration

        end = self.splits[self.current_idx]
        _tokens = self.tensor_dict["seq_length"]

        batch = {
            key: val[self.current_idx : end]
            for key, val in self.tensor_dict.items()
            if key not in self.to_ignore
        }
        clip_len = min(_tokens[end - 1], self.memory_estimator.max_seq_length())
        next_batch = {key: val[:, :clip_len] for key, val in batch.items()}

        self.current_idx += 1

        for fn in self._to_map:
            next_batch = fn(next_batch)

        return next_batch

    def find_optimal_splits(self):
        splits = []
        i = 0
        doubling_factor = 2
        max_doubling_attempts, max_steps = 8, 8
        dynamic_step_size = self.batch_size
        decreasing_attempts = 0

        num_tokens = self.tensor_dict["seq_length"]

        while i < len(num_tokens):
            best_fit_e_ind = i + self.batch_size  # Initialize to at least initial_batch_size

            # Try aggressive doubling first
            for doubling_i in range(max_doubling_attempts):
                tentative_e_ind = i + best_fit_e_ind * doubling_factor  # Double the last best fit
                tentative_e_ind = min(tentative_e_ind, len(num_tokens))
                max_token = int(num_tokens[tentative_e_ind - 1])
                est_memory = self.memory_estimator(max_token, int(tentative_e_ind - i))

                if est_memory <= self.memory_estimator.max_mem_gb:
                    best_fit_e_ind = tentative_e_ind
                else:
                    max_doubling_attempts = doubling_i  # Reduce max doubling attempts
                    break  # Exit loop if we exceed memory limit

            for _ in range(max_steps):
                tentative_e_ind = best_fit_e_ind + dynamic_step_size  # Add dynamic step size
                tentative_e_ind = min(tentative_e_ind, len(num_tokens))
                max_token = int(num_tokens[tentative_e_ind - 1])
                est_memory = self.memory_estimator(max_token, int(tentative_e_ind - i))

                if est_memory <= self.memory_estimator.max_mem_gb:
                    best_fit_e_ind = tentative_e_ind
                    break
                else:
                    dynamic_step_size //= 2  # halve the step size
                    decreasing_attempts += 1

            splits.append(best_fit_e_ind)
            i = best_fit_e_ind  # Move to the next batch

        return splits
