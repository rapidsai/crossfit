import torch
from itertools import islice

from crossfit.data.dataframe.dispatch import CrossFrame


class InMemoryDataLoader:
    def __init__(self, data_dict, batch_size):
        self.data_dict = data_dict
        self._batch_size = batch_size
        self.num_rows = len(next(iter(data_dict.values())))
        self.current_idx = 0

    @classmethod
    def from_df(cls, dataframe, *args, **kwargs):
        data_dict = CrossFrame(dataframe).convert(torch.Tensor)

        return cls(data_dict, *args, **kwargs)

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.num_rows:
            raise StopIteration

        batch_size = self.batch_size
        end = batch_size + self.current_idx

        next_batch = {key: val[self.current_idx : end] for key, val in self.data_dict.items()}

        self.current_idx += self.batch_size

        return next_batch

    def get_batches(self, n):
        return list(islice(self, n))
