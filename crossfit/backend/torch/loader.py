from typing import Dict, overload
from itertools import islice

import torch

from crossfit.backend.torch.model import Model
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.data.array.dispatch import crossarray
from crossfit.data.array.conversion import convert_array


class InMemoryLoader:
    @overload
    def __init__(
        self, data: Dict[str, torch.Tensor], batch_size: int, progress_bar=None
    ):
        ...

    @overload
    def __init__(self, data: CrossFrame, batch_size: int, progress_bar=None):
        ...

    def __init__(self, data, batch_size: int, progress_bar=None, max_seq_len=None):
        self.data = CrossFrame(data).cast(torch.Tensor)
        self.tensor_dict = self.data.to_dict()
        self._batch_size = batch_size
        self.num_rows = len(next(iter(self.tensor_dict.values())))
        self.current_idx = 0
        self._to_map = []
        self.progress_bar = progress_bar
        self.max_seq_len = max_seq_len

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

        batch = {
            key: val[self.current_idx : end] for key, val in self.tensor_dict.items()
        }
        if self.max_seq_len is not None:
            batch = {key: val[:, : self.max_seq_len] for key, val in batch.items()}

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
        initial_batch_size: int = 512,
        to_ignore=None,
        progress_bar=None,
    ):
        self.sort_key = sort_key
        self.to_ignore = to_ignore or []
        self.to_ignore.append("seq_length")
        self.model = model

        frame = CrossFrame(data).cast(torch.Tensor)
        seq_length = (frame[sort_key] != 0).sum(axis=1)
        self.sorted_indices = seq_length.argsort()
        frame = frame.apply(lambda x: x[self.sorted_indices])
        frame = frame.assign(seq_length=seq_length[self.sorted_indices])
        super().__init__(frame, initial_batch_size, progress_bar=progress_bar)

    def sort_column(self, col):
        indices = convert_array(self.sorted_indices, type(col))

        return col[indices]

    def sort_df(self, df):
        output = type(df)()
        for col in df.columns:
            output[col] = self.sort_column(df[col])

        return output

    def __next__(self):
        if self.current_idx >= self.num_rows:
            self.current_idx = 0
            raise StopIteration

        batch_size = 1

        def batch_seq_len(batch_size):
            end = self.current_idx + batch_size
            return int(
                min(
                    self.tensor_dict["seq_length"][end - 1], self.model.max_seq_length()
                )
            )

        while (
            self.current_idx + batch_size
        ) < self.num_rows and self.model.estimate_memory(
            batch_seq_len(batch_size), batch_size
        ) < (
            (self.model.max_mem_gb * 1024)
        ):
            batch_size += 1

        end = batch_size + self.current_idx

        batch = {
            key: val[self.current_idx : end]
            for key, val in self.tensor_dict.items()
            if key not in self.to_ignore
        }
        max_seq_len = batch_seq_len(batch_size)
        batch = {key: val[:, :max_seq_len] for key, val in batch.items()}

        self.current_idx += batch_size

        for fn in self._to_map:
            batch = fn(batch)

        if self.progress_bar is not None:
            self.progress_bar.update(batch_size)

        return batch
