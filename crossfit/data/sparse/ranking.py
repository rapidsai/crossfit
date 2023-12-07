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

import warnings

import numpy as np

from crossfit.data.array.dispatch import crossarray
from crossfit.data.array.masked import MaskedArray
from crossfit.data.sparse.dispatch import CrossSparse, SparseMatrixProtocol


class SparseLabels:
    def __init__(self, labels):
        self._labels: SparseMatrixProtocol = labels

        if self.binary:
            if not self.labels.is_binary():
                raise ValueError("Matrix may only contain 0 and 1 entries.")

        if self.labels.contains_inf():
            raise ValueError("Input contains NaN or Inf entries")

    def get_labels_for(self, ranking: "SparseRankings", k=None) -> MaskedArray:
        n_label_rows = self._labels.shape[0]
        n_ranking_rows = len(ranking)

        if n_ranking_rows < n_label_rows:
            raise ValueError(
                f"Gold labels contain {n_label_rows} rows, but input rankings only have"
                f" {n_ranking_rows} rows"
            )

        indices = ranking.rank_top_k(k)
        retrieved = self._labels.lookup(indices.data)

        return MaskedArray(retrieved, indices.mask)

    def as_rankings(self):
        return SparseRankings.from_scores(self._labels.tocsr(copy=True), warn_empty=False)

    @property
    def labels(self) -> SparseMatrixProtocol:
        return self._labels

    @classmethod
    def from_positive_indices(cls, indices):
        """
        Construct a binary labels instance from sparse data where only positive items are specified

        Parameters
        ----------
        indices : array_like, one row per context (e.g., user or query)
            Specifies positive indices for each sample. Must be 1D or 2D, but row lengths can
            differ.

        Raises
        ------
        ValueError
            if `indices` is of invalid shape, type or contains duplicate, negative or non-integer
            indices.

        Examples
        --------
        >>> BinaryLabels.from_positive_indices([[1,2], [2]])
        <rankereval.data.BinaryLabels...>
        """
        return cls(CrossSparse.from_nonzero_indices(indices))

    @classmethod
    def from_matrix(cls, labels):
        """
        Construct a binary labels instance from dense or sparse matrix where each item's label is
        specified.

        Parameters
        ----------
        labels : 1D or 2D array, one row per context (e.g., user or query)
            Contains binary labels for each item. Labels must be in {0, 1}.

        Raises
        ------
        ValueError
            if `labels` is of invalid shape, type or non-binary.

        Examples
        --------
        >>> BinaryLabels.from_matrix([[0, 1, 1], [0, 0, 1]])
        <rankereval.data.BinaryLabels...>
        """
        return cls(CrossSparse.from_matrix(labels))

    def labels_to_list(self):
        return self._labels.tolil().data.tolist()

    def indices_to_list(self):
        return self._labels.tolil()

    def get_n_positives(self, n_rankings):
        n_label_rows = self.labels.shape[0]
        n_pos = self.labels.getnnz(axis=1)
        if n_label_rows == 1:
            with crossarray:
                n_pos = np.tile(n_pos, n_rankings)

        return n_pos

    def __str__(self):
        return str(self.indices_to_list())


class InvalidValuesWarning(UserWarning):
    pass


class SparseBinaryLabels(SparseLabels):
    """
    Represents binary ground truth data (e.g., 1 indicating relevance).
    """

    binary = True


class SparseNumericLabels(SparseBinaryLabels):
    """
    Represents numeric ground truth data (e.g., relevance labels from 1-5).
    """

    binary = False


class Rankings:
    """
    Data structure where rankings have the same length (approximately).
    """

    def __init__(self, indices, mask=None, warn_empty=True):
        if warn_empty:
            with crossarray:
                n_empty_rows = ((~mask).sum(axis=1) == 0).sum()
            if n_empty_rows:
                warnings.warn(
                    f"Input rankings have {n_empty_rows} empty rankings (rows). "
                    + "These will impact the mean scores."
                    + str(indices),
                    InvalidValuesWarning,
                )
        self.indices = indices
        self.mask = mask

    @classmethod
    def _verify_input(cls, arr, dtype=np.floating):
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input needs to be a numpy matrix")
        arr = np.asarray(np.atleast_2d(arr))
        if arr.ndim != 2:
            raise ValueError("Input arrays need to be 1D or 2D.")
        elif not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"Input array needs to be of type {dtype}")

        if np.issubdtype(dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                warnings.warn(
                    "Input contains NaN or Inf entries which will be ignored.",
                    InvalidValuesWarning,
                )
                arr[~np.isfinite(arr)] = np.NINF
        elif not np.issubdtype(dtype, np.integer):
            raise TypeError("dtype argument must be floating or int")
        return arr

    @classmethod
    def from_ranked_indices(cls, indices, valid_items=None, invalid_items=None):
        """
        Set indices to -1 (or any other negative value) to indicate invalid index
        """
        indices = cls._verify_input(indices, dtype=np.integer)

        if valid_items is not None or invalid_items is not None:
            raise NotImplementedError("Not implemented yet")
        mask = indices < 0
        return cls(indices, mask)

    @classmethod
    def from_scores(
        cls,
        raw_scores,
        valid_items=None,
        invalid_items=None,
        warn_empty=True,
        k_max=None,
    ):
        raw_scores = cls._verify_input(raw_scores, dtype=np.floating)

        if valid_items is not None:
            invalid_idx = CrossSparse.from_nonzero_indices(invalid_items).csr.toarray() == 0
            raw_scores -= np.inf * invalid_idx
        if invalid_items is not None:
            invalid_items = CrossSparse.from_nonzero_indices(invalid_items).csr
            raw_scores -= np.inf * invalid_items

        mask = ~np.isfinite(raw_scores)
        if k_max is None:
            sorted_idx = np.argsort(-raw_scores, axis=1, kind="stable")
        else:
            sorted_idx = topk(raw_scores, k_max)
        mask = np.take_along_axis(mask, sorted_idx, axis=1)
        return cls(sorted_idx, mask)

    def rank_top_k(self, k=None) -> MaskedArray:
        if k is None:
            k = self.shape[1]
        indices = self.indices[:, :k]
        mask = self.mask[:, :k]

        return MaskedArray(indices, mask)

    def to_list(self):
        return self.indices.tolist()

    def __str__(self):
        return str(self.indices)

    def __len__(self):
        return self.indices.shape[0]


class SparseRankings(Rankings):
    """
    Represents (predicted) rankings to be evaluated.
    """

    def __init__(self, indices, valid_items=None, invalid_items=None, warn_empty=True):
        if valid_items is not None:
            valid_items = CrossSparse.from_nonzero_indices(valid_items)
            indices.intersection(valid_items)
        if invalid_items is not None:
            invalid_items = CrossSparse.from_nonzero_indices(invalid_items)
            indices.difference(invalid_items)
        if not indices.isfinite():
            warnings.warn(
                "Input contains NaN or Inf entries which will be ignored.",
                InvalidValuesWarning,
            )
            indices.remove_infinite()
        n_empty_rows = indices.count_empty_rows()
        if n_empty_rows and warn_empty:
            warnings.warn(
                f"Input rankings have {n_empty_rows} empty rankings (rows). "
                + "These will impact the mean scores."
                + str(indices.todense()),
                InvalidValuesWarning,
            )
        self.indices: SparseMatrixProtocol = indices

    @classmethod
    def from_ranked_indices(cls, indices, valid_items=None, invalid_items=None):
        """
        Construct a rankings instance from data where item indices are specified in ranked order.

        Parameters
        ----------
        indices : array_like, one row per ranking
            Indices of items after ranking. Must be 1D or 2D, but row lengths can differ.
        valid_items : array_like, one row per ranking
            Indices of valid items (e.g., candidate set). Invalid items will be discarded from
            ranking.

        Raises
        ------
        ValueError
            if `indices` or `valid_items` of invalid shape or type.

        Examples
        --------
        >>> Rankings.from_ranked_indices([[5, 2], [4, 3, 1]])
        <rankereval.data.Rankings...>
        """
        indices = CrossSparse.from_lil(indices)
        return cls(indices, valid_items, invalid_items)

    @classmethod
    def from_scores(cls, raw_scores, valid_items=None, invalid_items=None, warn_empty=True):
        """
        Construct a rankings instance from raw scores where each item's score is specified.
        Items will be ranked in descending order (higher scores meaning better).

        Parameters
        ----------
        raw_scores : array_like, one row per ranking
            Contains raw scores for each item. Must be 1D or 2D, but row lengths can differ.
        valid_items : array_like, one row per ranking
            Indices of valid items (e.g., candidate set). Invalid items will be discarded from
            ranking.

        Raises
        ------
        ValueError
            if `raw_scores` or `valid_items` of invalid shape or type.

        Warns
        ------
        InvalidValuesWarning
            if `raw_scores` contains non-finite values.

        Examples
        --------
        >>> Rankings.from_scores([[0.1, 0.5, 0.2], [0.4, 0.2, 0.5]])
        <rankereval.data.Rankings...>
        """
        indices = CrossSparse.from_values(raw_scores, keep_zeros=True)
        indices.sort()

        return cls(indices, valid_items, invalid_items, warn_empty=warn_empty)

    def rank_top_k(self, k=None):
        return self.indices.rank_top_k(k)

    def to_list(self):
        return self.indices.tolil()


def topk(x, k, return_scores=False):
    with crossarray:
        # partition into k largest elements first
        index_array = np.sort(np.argpartition(-x, kth=k - 1, axis=-1)[:, :k])
        top_k_partition = np.take_along_axis(x, index_array, axis=-1)

        # stable argsort in descending order
        top_idx_local = top_k_partition.shape[1] - 1
        top_idx_local -= np.fliplr(np.argsort(np.fliplr(top_k_partition), axis=-1, kind="stable"))

        # sort the top partition
        top_idx = np.take_along_axis(index_array, top_idx_local, axis=-1)
        if not return_scores:
            return top_idx
        else:
            top_scores = np.take_along_axis(top_k_partition, top_idx_local, axis=-1)

            return top_scores, top_idx
