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

from typing import Protocol

import numpy as np

from crossfit.data.array.dispatch import crossarray
from crossfit.data.array.masked import MaskedArray


class SparseMatrixProtocol(Protocol):
    @classmethod
    def from_values(cls, data, keep_zeros=False):
        ...

    @classmethod
    def from_nonzero_indices(cls, indices):
        ...

    @classmethod
    def from_matrix(cls, matrix, keep_zeros=False):
        ...

    @classmethod
    def from_lil(cls, rows, data=None, dtype="float32", keep_zeros=False):
        ...

    def sort(self):
        ...

    def intersection(self, other):
        ...

    def difference(self, other):
        ...

    def isfinite(self):
        ...

    def remove_infinite(self):
        ...

    def eliminate_zeros(self):
        ...

    def todense(self):
        ...

    def todense_masked(self, shape) -> MaskedArray:
        ...

    def lookup(self, indices):
        ...

    def max_nnz_row_values(self):
        ...

    def count_empty_rows(self):
        ...

    def tolil(self):
        ...

    def rank_top_k(self, k=None) -> MaskedArray:
        ...

    def __str__(self):
        ...

    def is_binary(self) -> bool:
        ...

    def contains_inf(self) -> bool:
        ...


class SparseMatrixBackend:
    """
    Stores sparse matrix data in unsorted CSR format (i.e., column indices in each row are
    unsorted).
    """

    def __init__(self, idx_ptr, col_idx, data, shape=None):
        self.idx_ptr = idx_ptr
        self.col_idx = col_idx
        self.data = data
        self.shape = shape

    @classmethod
    def from_values(cls, data, keep_zeros=False):
        raise NotImplementedError()

    @classmethod
    def from_nonzero_indices(cls, indices):
        raise NotImplementedError()

    @classmethod
    def from_matrix(cls, matrix, keep_zeros=False):
        raise NotImplementedError()

    @classmethod
    def from_lil(cls, rows, data=None, dtype="float32", keep_zeros=False):
        raise NotImplementedError()

    def sort(self):
        raise NotImplementedError()

    def intersection(self, other):
        raise NotImplementedError()

    def difference(self, other):
        raise NotImplementedError()

    def isfinite(self):
        raise NotImplementedError()

    def remove_infinite(self):
        raise NotImplementedError()

    def eliminate_zeros(self):
        raise NotImplementedError()

    def todense(self):
        raise NotImplementedError()

    def todense_masked(self, shape) -> MaskedArray:
        raise NotImplementedError()

    def lookup(self, indices):
        raise NotImplementedError()

    def rank_top_k(self, k=None) -> MaskedArray:
        raise NotImplementedError()

    def max_nnz_row_values(self):
        """Returns maximum number of non-zero entries in any row."""
        return (self.idx_ptr[1:] - self.idx_ptr[:-1]).max()

    def count_empty_rows(self):
        return ((self.idx_ptr[1:] - self.idx_ptr[:-1]) == 0).sum()

    def tolil(self):
        res = []
        for i in range(len(self.idx_ptr) - 1):
            start, end = self.idx_ptr[i], self.idx_ptr[i + 1]
            res += [self.col_idx[start:end].tolist()]
        return res

    def is_binary(self) -> bool:
        with crossarray:
            return np.all(np.isin(self.data, [0, 1]))

    def contains_inf(self) -> bool:
        with crossarray:
            nonfinite_entries = ~np.isfinite(self.data)
            return np.any(nonfinite_entries)

    def to_pytrec(self, is_run=False):
        sparse_matrix = self.tocsr()

        qrel = {}
        for i in range(self.indices.shape[0]):
            query_id = f"q{i+1}"
            qrel[query_id] = {}

            row = sparse_matrix[i]
            for j, score in zip(row.indices, row.data):
                doc_id = f"d{j+1}"
                qrel[query_id][doc_id] = int(score) if is_run else float(score)

        return qrel

    def __str__(self):
        return str((self.idx_ptr, self.col_idx, self.data))
