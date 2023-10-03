import itertools

import cupy as cp
import cupyx.scipy.sparse as sp
import numba

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

    def tocsr(self):
        return sp.csr_matrix((self.data, self.col_idx, self.idx_ptr), copy=False, shape=self.shape)

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
        if self.shape[0] != other.shape[0]:
            raise ValueError("Matrices need to have the same number of rows!")
        blocks, threads = determine_blocks_threads(len(self.idx_ptr) - 1)
        _numba_setop[blocks, threads](
            self.idx_ptr, self.col_idx, self.data, other.idx_ptr, other.col_idx, mode
        )
        self.eliminate_zeros()

    def sort(self):
        blocks, threads = determine_blocks_threads(len(self.idx_ptr))
        _numba_sort[blocks, threads](self.idx_ptr, self.col_idx, self.data)

    def intersection(self, other):
        self._setop(other, True)

    def difference(self, other):
        self._setop(other, False)

    def isfinite(self):
        return cp.all(cp.isfinite(self.data))

    def todense_masked(self, shape) -> MaskedArray:
        condensed = cp.zeros(shape, dtype=self.col_idx.dtype)
        mask = cp.ones(shape, dtype=cp.bool_)

        blocks, threads = determine_blocks_threads(len(self.idx_ptr) - 1)
        _numba_csr_to_dense_masked[blocks, threads](
            self.idx_ptr, self.col_idx, condensed, mask, shape
        )

        data = cp.asarray(condensed)

        return MaskedArray(data, mask)

    def lookup(self, indices):
        vals = cp.zeros_like(indices)
        blocks, threads = determine_blocks_threads(indices.shape[0])
        _numba_lookup[blocks, threads](self.idx_ptr, self.col_idx, self.data, indices, vals)
        return cp.asarray(vals)

    def rank_top_k(self, k=None) -> MaskedArray:
        if k is None:
            k = self.indices.max_nnz_row_values()
        return self.todense_masked((self.shape[0], k))

    def getnnz(self, axis=None):
        return self.tocsr().getnnz(axis=axis)


@CrossSparse.register(cp.ndarray)
@CrossSparse.register(list)
def _cupy_sparse(data):
    return CPSparseMatrixBackend


@CrossSparse.register_lazy("scipy")
def _scipy_sparse():
    import cupyx.scipy.sparse as sp

    @CrossSparse.register(sp.csr_matrix)
    @CrossSparse.register(sp.coo_matrix)
    def _pd_frame(data):
        return CPSparseMatrixBackend


@numba.cuda.jit(device=True)
def cuda_searchsorted(arr, val, side):
    """
    Binary search on a sorted array.

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    0       ``arr[i-1] < val <= arr[i]``
    1       ``arr[i-1] <= val < arr[i]``
    ======  ============================
    """
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < val or (side == 1 and arr[mid] <= val):
            left = mid + 1
        else:
            right = mid
    return left


@numba.cuda.jit
def _numba_lookup(A_indptr, A_cols, A_data, B, vals):
    i = numba.cuda.grid(1)

    n_rows_a = len(A_indptr) - 1
    if n_rows_a == len(B):
        ind_start, ind_end = A_indptr[i], A_indptr[i + 1]
        for j in range(B.shape[1]):
            left_idx = cuda_searchsorted(A_cols[ind_start:ind_end], B[i][j], 0)
            right_idx = cuda_searchsorted(A_cols[ind_start:ind_end], B[i][j], 1)
            if left_idx != right_idx:
                vals[i][j] = A_data[ind_start:ind_end][left_idx]
    else:
        for j in range(B.shape[1]):
            left_idx = cuda_searchsorted(A_cols, B[i][j], 0)
            right_idx = cuda_searchsorted(A_cols, B[i][j], 1)
            if left_idx != right_idx:
                vals[i][j] = A_data[left_idx]


@numba.cuda.jit
def _numba_sort(idx_ptr, col_idx, data):
    i = numba.cuda.grid(1)

    if i < len(idx_ptr) - 1:
        start, end = idx_ptr[i], idx_ptr[i + 1]

        # Custom insertion sort for simplicity.
        # This may not be the most efficient for long lists.
        for j in range(start + 1, end):
            key_data = data[j]
            key_col_idx = col_idx[j]
            k = j - 1
            while k >= start and data[k] < key_data:
                data[k + 1] = data[k]
                col_idx[k + 1] = col_idx[k]
                k -= 1
            data[k + 1] = key_data
            col_idx[k + 1] = key_col_idx


@numba.cuda.jit
def _numba_setop(self_idx_ptr, self_col_idx, self_data, other_idx_ptr, other_col_idx, intersect):
    i = numba.cuda.grid(1)

    if i < len(self_idx_ptr) - 1:
        ss, se = self_idx_ptr[i], self_idx_ptr[i + 1]
        os, oe = other_idx_ptr[i], other_idx_ptr[i + 1]

        for j in range(ss, se):
            left_idx = cuda_searchsorted(other_col_idx[os:oe], self_col_idx[j], 0)
            right_idx = cuda_searchsorted(other_col_idx[os:oe], self_col_idx[j], 1)

            if intersect:
                found = left_idx == right_idx
            else:
                found = left_idx != right_idx

            if found:
                self_data[j] = 0


@numba.cuda.jit
def _numba_csr_to_dense_masked(idx_ptr, col_idx, condensed, mask, shape):
    i = numba.cuda.grid(1)

    if i < len(idx_ptr) - 1:
        start, end = idx_ptr[i], idx_ptr[i + 1]
        length = min(end - start, shape[1])
        for j in range(length):
            condensed[i][j] = col_idx[start + j]
            mask[i][j] = False


def determine_blocks_threads(n, threads_per_block=32):
    number_of_blocks = (n + threads_per_block - 1) // threads_per_block
    return number_of_blocks, threads_per_block
