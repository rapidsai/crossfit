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

import cupy as cp
import cupyx.scipy.sparse as sp

from crossfit.data.array.masked import MaskedArray
from crossfit.data.sparse.core import SparseMatrixBackend
from crossfit.data.sparse.dispatch import CrossSparse


class CPSparseMatrixBackend(SparseMatrixBackend):
    def __init__(self, idx_ptr: cp.ndarray, col_idx: cp.ndarray, data: cp.ndarray, shape=None):
        if shape is None:
            if len(col_idx):
                M = col_idx.max() + 1
            else:
                M = 0
            shape = (len(idx_ptr) - 1, M)

        super().__init__(idx_ptr.copy(), col_idx.copy(), data.copy(), shape)

    @classmethod
    def supports(cls):
        import cupyx.scipy.sparse as sp

        return [cp.ndarray, sp.csr_matrix, sp.coo_matrix]

    @classmethod
    def from_values(cls, data, keep_zeros=False):
        if isinstance(data, list):
            if len(data) == 0 or cp.ndim(data[0]) == 0:
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
        if isinstance(matrix, cp.ndarray) or isinstance(matrix, list):
            if isinstance(matrix, list):
                matrix = cp.asarray(matrix, dtype=object).astype(cp.float32)
            matrix = cp.atleast_2d(matrix)
            if not cp.issubdtype(matrix.dtype, cp.number) or cp.issubdtype(matrix.dtype, cp.bool_):
                raise ValueError("Input must be numeric")
            elif matrix.ndim != 2:
                raise ValueError("Input arrays need to be 1D or 2D.")
            if keep_zeros:
                matrix += 1 - matrix[cp.isfinite(matrix)].min()
            x = sp.csr_matrix(matrix)
            if not keep_zeros:
                x.eliminate_zeros()
        elif sp.issparse(matrix):
            x = matrix.tocsr()
        else:
            raise ValueError("Input type not supported.")
        return cls(x.indptr, x.indices, x.data, x.shape)

    @classmethod
    def from_lil(cls, rows, data=None, dtype=cp.float32, keep_zeros=False):
        if not isinstance(rows, list) and not isinstance(rows, cp.ndarray):
            raise ValueError("Invalid input type.")
        if len(rows) == 0 or cp.ndim(rows[0]) == 0:
            rows = [rows]
        idx_ptr = cp.asarray([0] + [len(x) for x in rows], dtype=int).cumsum()
        try:
            col_idx = cp.fromiter(
                itertools.chain.from_iterable(rows), dtype=int, count=idx_ptr[-1].item()
            )
            if data is None:
                data = cp.ones_like(col_idx, dtype=dtype)
            else:
                data = cp.fromiter(
                    itertools.chain.from_iterable(data), dtype=dtype, count=idx_ptr[-1].item()
                )
                if keep_zeros:
                    data += 1 - data[cp.isfinite(data)].min()
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
        return cp.asarray(self.tocsr().todense())

    def remove_infinite(self):
        if not self.isfinite():
            self.data[~cp.isfinite(self.data)] = 0
            self.eliminate_zeros()

    def eliminate_zeros(self):
        csr = self.tocsr()
        csr.eliminate_zeros()
        self.data, self.col_idx, self.idx_ptr = csr.data, csr.indices, csr.indptr

    def _setop(self, other, mode):
        from crossfit.backend.cupy.kernels import _numba_setop, determine_blocks_threads

        if self.shape[0] != other.shape[0]:
            raise ValueError("Matrices need to have the same number of rows!")
        blocks, threads = determine_blocks_threads(len(self.idx_ptr) - 1)
        _numba_setop[blocks, threads](
            self.idx_ptr, self.col_idx, self.data, other.idx_ptr, other.col_idx, mode
        )
        self.eliminate_zeros()

    def sort(self):
        from crossfit.backend.cupy.kernels import _numba_sort, determine_blocks_threads

        blocks, threads = determine_blocks_threads(len(self.idx_ptr) - 1)
        _numba_sort[blocks, threads](self.idx_ptr, self.col_idx, self.data)

    def intersection(self, other):
        self._setop(other, True)

    def difference(self, other):
        self._setop(other, False)

    def isfinite(self):
        return cp.all(cp.isfinite(self.data))

    def todense_masked(self, shape) -> MaskedArray:
        from crossfit.backend.cupy.kernels import (
            _numba_csr_to_dense_masked,
            determine_blocks_threads,
        )

        condensed = cp.zeros(shape, dtype=self.col_idx.dtype)
        mask = cp.ones(shape, dtype=cp.bool_)

        blocks, threads = determine_blocks_threads(len(self.idx_ptr) - 1)
        _numba_csr_to_dense_masked[blocks, threads](
            self.idx_ptr, self.col_idx, condensed, mask, shape
        )

        data = cp.asarray(condensed)

        return MaskedArray(data, mask)

    def lookup(self, indices):
        from crossfit.backend.cupy.kernels import _numba_lookup, determine_blocks_threads

        vals = cp.zeros_like(indices)
        blocks, threads = determine_blocks_threads(indices.shape[0])
        _numba_lookup[blocks, threads](self.idx_ptr, self.col_idx, self.data, indices, vals)
        return cp.asarray(vals)

    def rank_top_k(self, k=None) -> MaskedArray:
        if k is None:
            k = self.indices.max_nnz_row_values()
        return self.todense_masked((self.shape[0], k))

    def getnnz(self, axis=None):
        csr_mat = self.tocsr()

        if axis is None:
            return csr_mat.getnnz()
        elif axis == 0:
            # Count non-zero elements along axis 0 (columns)
            return cp.diff(csr_mat.indptr).sum(axis=axis)
        elif axis == 1:
            # Count non-zero elements along axis 1 (rows)
            return cp.diff(csr_mat.indptr)
        else:
            raise ValueError("Invalid axis, expected None, 0 or 1")

    def is_binary(self) -> bool:
        return cp.all(cp.isin(self.data, cp.asarray([0, 1])))

    def contains_inf(self) -> bool:
        nonfinite_entries = ~cp.isfinite(self.data)
        return cp.any(nonfinite_entries)


@CrossSparse.register(cp.ndarray)
def _cupy_sparse(data):
    return CPSparseMatrixBackend


@CrossSparse.register_lazy("scipy")
def _scipy_sparse():
    import cupyx.scipy.sparse as sp

    @CrossSparse.register(sp.csr_matrix)
    @CrossSparse.register(sp.coo_matrix)
    def _pd_frame(data):
        return CPSparseMatrixBackend
