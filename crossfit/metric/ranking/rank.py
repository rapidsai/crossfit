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

import numpy as np

from crossfit.data.array.masked import MaskedArray
from crossfit.metric.ranking.base import BinaryRankingMetric, SparseBinaryLabels


class ReciprocalRank(BinaryRankingMetric):
    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        # It is 1/rank if document appears in top k, 0 otherwise
        scores = np.max(labels / ranks, axis=-1, initial=0.0)
        scores[n_pos == 0] = np.nan  # Not defined for no multiple positives

        return scores


class MeanRanks(BinaryRankingMetric):
    def __init__(self):
        self._k = None

    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels.filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        scores = np.sum(ranks * labels, axis=-1)
        scores[n_pos > 0] = scores[n_pos > 0] / n_pos[n_pos > 0]
        scores[n_pos == 0] = np.nan
        return scores


class FirstRelevantRank(BinaryRankingMetric):
    def __init__(self):
        self._k = None

    def _score(self, y_true: SparseBinaryLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels.filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        ranks = ranks * labels
        ranks[ranks == 0] = np.inf
        scores = np.min(ranks, axis=-1)
        scores[n_pos == 0] = np.nan

        return scores
