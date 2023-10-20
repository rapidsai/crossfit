from numba import cuda


@cuda.jit(device=True)
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


@cuda.jit
def _numba_lookup(A_indptr, A_cols, A_data, B, vals):
    i = cuda.grid(1)

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
def _numba_setop(
    self_idx_ptr, self_col_idx, self_data, other_idx_ptr, other_col_idx, intersect
):
    i = cuda.grid(1)

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
