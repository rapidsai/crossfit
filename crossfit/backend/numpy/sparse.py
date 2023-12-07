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

import itertools

import numba
import numpy as np
import scipy.sparse as sp

from crossfit.data.array.masked import MaskedArray
from crossfit.data.sparse.core import SparseMatrixBackend
from crossfit.data.sparse.dispatch import CrossSparse


class NPSparseMatrixBackend(SparseMatrixBackend):
    def __init__(self, idx_ptr: np.ndarray, col_idx: np.ndarray, data: np.ndarray, shape=None):
        if shape is None:
            if len(col_idx):
                M = col_idx.max() + 1
            else:
                M = 0
            shape = (len(idx_ptr) - 1, M)

        super().__init__(idx_ptr.copy(), col_idx.copy(), data.copy(), shape)

    @classmethod
    def supports(cls):
        return [list, np.ndarray, sp.csr_matrix, sp.coo_matrix]

    @classmethod
    def from_values(cls, data, keep_zeros=False):
        if isinstance(data, list):
            if len(data) == 0 or np.ndim(data[0]) == 0:
                data = [data]
            idx = [list(range(len(r))) for r in data]
            return cls.from_lil(idx, data, keep_zeros=keep_zeros)
        else:
            return cls.from_matrix(data, keep_zeros=keep_zeros)

    @classmethod
    def from_nonzero_indices(cls, indices):
        if sp.issparse(indices):
            x = indices.tocsr()
            return cls(x.indptr, x.indices, x.data, x.shape)
        else:
            return cls.from_lil(indices)

    @classmethod
    def from_matrix(cls, matrix, keep_zeros=False):
        if isinstance(matrix, np.ndarray) or isinstance(matrix, list):
            if isinstance(matrix, list):
                matrix = np.asarray(matrix, dtype=object).astype(np.float32)
            matrix = np.atleast_2d(matrix)
            if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.bool_):
                raise ValueError("Input must be numeric")
            elif matrix.ndim != 2:
                raise ValueError("Input arrays need to be 1D or 2D.")
            if keep_zeros:
                matrix += 1 - matrix[np.isfinite(matrix)].min()
            x = sp.csr_matrix(matrix)
            if not keep_zeros:
                x.eliminate_zeros()
        elif sp.issparse(matrix):
            x = matrix.tocsr()
        else:
            raise ValueError("Input type not supported.")
        return cls(x.indptr, x.indices, x.data, x.shape)

    @classmethod
    def from_lil(cls, rows, data=None, dtype=np.float32, keep_zeros=False):
        if not isinstance(rows, list) and not isinstance(rows, np.ndarray):
            raise ValueError("Invalid input type.")
        if len(rows) == 0 or np.ndim(rows[0]) == 0:
            rows = [rows]
        idx_ptr = np.asarray([0] + [len(x) for x in rows], dtype=int).cumsum()
        try:
            col_idx = np.fromiter(itertools.chain.from_iterable(rows), dtype=int, count=idx_ptr[-1])
            if data is None:
                data = np.ones_like(col_idx, dtype=dtype)
            else:
                data = np.fromiter(
                    itertools.chain.from_iterable(data), dtype=dtype, count=idx_ptr[-1]
                )
                if keep_zeros:
                    data += 1 - data[np.isfinite(data)].min()
        except TypeError:
            raise ValueError("Invalid values in input.")
        if len(data) != len(col_idx):
            raise ValueError("rows and data need to have same length")
        instance = cls(idx_ptr, col_idx, data)
        if not keep_zeros:
            instance.eliminate_zeros()
        return instance

    def tocsr(self, copy=False):
        return sp.csr_matrix((self.data, self.col_idx, self.idx_ptr), copy=copy, shape=self.shape)

    def todense(self):
        return np.asarray(self.tocsr().todense())

    def remove_infinite(self):
        if not self.isfinite():
            self.data[~np.isfinite(self.data)] = 0
            self.eliminate_zeros()

    def eliminate_zeros(self):
        csr = self.tocsr()
        csr.eliminate_zeros()
        self.data, self.col_idx, self.idx_ptr = csr.data, csr.indices, csr.indptr

    def _setop(self, other, mode):
        if self.shape[0] != other.shape[0]:
            raise ValueError("Matrices need to have the same number of rows!")
        _numba_setop(self.idx_ptr, self.col_idx, self.data, other.idx_ptr, other.col_idx, mode)
        self.eliminate_zeros()

    def sort(self):
        _numba_sort(self.idx_ptr, self.col_idx, self.data)

    def intersection(self, other):
        self._setop(other, True)

    def difference(self, other):
        self._setop(other, False)

    def isfinite(self):
        return np.all(np.isfinite(self.data))

    def todense_masked(self, shape) -> MaskedArray:
        data, mask = _numba_csr_to_dense_masked(self.idx_ptr, self.col_idx, shape)

        return MaskedArray(data, mask)

    def lookup(self, indices):
        return _numba_lookup(self.idx_ptr, self.col_idx, self.data, indices)

    def rank_top_k(self, k=None) -> MaskedArray:
        if k is None:
            k = self.max_nnz_row_values()
        return self.todense_masked((self.shape[0], k))

    def getnnz(self, axis=None):
        return self.tocsr().getnnz(axis=axis)


@CrossSparse.register(np.ndarray)
@CrossSparse.register(list)
def _numpy_sparse(data):
    return NPSparseMatrixBackend


@CrossSparse.register_lazy("scipy")
def _scipy_sparse():
    import scipy.sparse as sp

    @CrossSparse.register(sp.csr_matrix)
    @CrossSparse.register(sp.coo_matrix)
    def _pd_frame(data):
        return NPSparseMatrixBackend


@numba.njit(parallel=True)
def _numba_lookup(A_indptr, A_cols, A_data, B):
    """
    Numba accelerated version of lookup table
    """
    # Non-existing indices are assigned label of 0.0
    vals = np.zeros(B.shape, dtype=np.float32)

    n_rows_a = len(A_indptr) - 1
    if n_rows_a == len(B):
        for i in numba.prange(B.shape[0]):
            ind_start, ind_end = A_indptr[i], A_indptr[i + 1]
            for j in range(len(B[i])):
                for k in range(ind_start, ind_end):
                    if A_cols[k] == B[i][j]:
                        vals[i][j] = A_data[k]
                        break
    else:
        for i in numba.prange(B.shape[0]):
            for j in range(len(B[i])):
                for k in range(len(A_cols)):
                    if A_cols[k] == B[i][j]:
                        vals[i][j] = A_data[k]
                        break

    return vals


@numba.njit(parallel=True)
def _numba_sort(idx_ptr, col_idx, data):
    for i in numba.prange(len(idx_ptr) - 1):
        start, end = idx_ptr[i], idx_ptr[i + 1]
        args = (-data[start:end]).argsort(kind="mergesort")
        data[start:end] = data[start:end][args]
        col_idx[start:end] = col_idx[start:end][args]


@numba.njit(parallel=True)
def _numba_setop(self_idx_ptr, self_col_idx, self_data, other_idx_ptr, other_col_idx, intersect):
    for i in numba.prange(len(self_idx_ptr) - 1):
        ss, se = self_idx_ptr[i], self_idx_ptr[i + 1]
        os, oe = other_idx_ptr[i], other_idx_ptr[i + 1]

        for j in range(ss, se):
            found = False
            for k in range(os, oe):
                if self_col_idx[j] == other_col_idx[k]:
                    found = True
                    break
            if (intersect and not found) or (not intersect and found):
                self_data[j] = 0


@numba.njit
def _numba_csr_to_dense_masked(idx_ptr, col_idx, shape):
    condensed = np.zeros(shape, dtype=col_idx.dtype)
    mask = np.ones(shape, dtype=np.bool_)
    for i in range(len(idx_ptr) - 1):
        start, end = idx_ptr[i], idx_ptr[i + 1]
        length = min(end - start, shape[1])
        condensed[i][:length] = col_idx[start : start + length]
        mask[i][:length] = False

    return condensed, mask
