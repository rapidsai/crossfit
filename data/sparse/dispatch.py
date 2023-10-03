from dask.utils import Dispatch

from crossfit.data.sparse.core import SparseMatrixProtocol


class _CrossSparseDispatch(Dispatch, SparseMatrixProtocol):
    def __call__(self, data):
        return super().__call__(data)

    @classmethod
    def from_values(cls, data, keep_zeros=False) -> SparseMatrixProtocol:
        cross_cls = CrossSparse(data)

        return cross_cls.from_matrix(data, keep_zeros=keep_zeros)

    @classmethod
    def from_nonzero_indices(cls, indices) -> SparseMatrixProtocol:
        cross_cls = CrossSparse(indices)

        return cross_cls.from_nonzero_indices(indices)

    @classmethod
    def from_matrix(cls, matrix, keep_zeros=False) -> SparseMatrixProtocol:
        cross_cls = CrossSparse(matrix)

        return cross_cls.from_matrix(matrix, keep_zeros=keep_zeros)

    @classmethod
    def from_lil(cls, rows, data=None, dtype="float32", keep_zeros=False) -> SparseMatrixProtocol:
        cross_cls = CrossSparse(rows)

        return cross_cls.from_lil(rows, data=data, dtype=dtype, keep_zeros=keep_zeros)


CrossSparse: SparseMatrixProtocol = _CrossSparseDispatch(name="sparse_dispatch")
