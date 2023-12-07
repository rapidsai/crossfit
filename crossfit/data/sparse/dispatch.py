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

from dask.utils import Dispatch

from crossfit.data.sparse.core import SparseMatrixProtocol


class _CrossSparseDispatch(Dispatch, SparseMatrixProtocol):
    def __call__(self, data):
        from crossfit.backend.cupy.sparse import CPSparseMatrixBackend
        from crossfit.backend.numpy.sparse import NPSparseMatrixBackend

        backends = [CPSparseMatrixBackend, NPSparseMatrixBackend]

        for backend in backends:
            for support in backend.supports():
                if isinstance(data, support):
                    return backend

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
