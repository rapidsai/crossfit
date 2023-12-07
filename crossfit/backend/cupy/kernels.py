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

from numba import cuda


@cuda.jit
def _numba_lookup(A_indptr, A_cols, A_data, B, vals):
    i = cuda.grid(1)

    if i < B.shape[0]:
        n_rows_a = len(A_indptr) - 1
        if n_rows_a == len(B):
            ind_start, ind_end = A_indptr[i], A_indptr[i + 1]
            for j in range(B.shape[1]):
                for k in range(ind_start, ind_end):
                    if A_cols[k] == B[i][j]:
                        vals[i][j] = A_data[k]
                        break
        else:
            for j in range(B.shape[1]):
                for k in range(len(A_cols)):
                    if A_cols[k] == B[i][j]:
                        vals[i][j] = A_data[k]
                        break


@cuda.jit
def _numba_sort(idx_ptr, col_idx, data):
    i = cuda.grid(1)

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


@cuda.jit
def _numba_setop(self_idx_ptr, self_col_idx, self_data, other_idx_ptr, other_col_idx, intersect):
    i = cuda.grid(1)

    if i < len(self_idx_ptr) - 1:
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


@cuda.jit
def _numba_csr_to_dense_masked(idx_ptr, col_idx, condensed, mask, shape):
    i = cuda.grid(1)

    if i < len(idx_ptr) - 1:
        start, end = idx_ptr[i], idx_ptr[i + 1]
        length = min(end - start, shape[1])
        for j in range(length):
            condensed[i][j] = col_idx[start + j]
            mask[i][j] = False


def determine_blocks_threads(n, threads_per_block=32):
    number_of_blocks = (n + threads_per_block - 1) // threads_per_block
    return number_of_blocks, threads_per_block
