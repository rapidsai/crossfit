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
from crossfit.metric.ranking.base import BinaryRankingMetric, SparseLabels


class Recall(BinaryRankingMetric):
    def __init__(self, k, truncated=False):
        super().__init__(k)
        self._truncated = truncated

    def _recall(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(
            (y_pred_labels.data[:, : self._k] >= 1) & (~y_pred_labels.mask[:, : self._k]),
            axis=-1,
        )

        scores = np.nan * np.zeros_like(n_relevant, dtype=float)
        if self._truncated:
            denominator = np.clip(n_pos[n_pos > 0], None, self._k)
        else:
            denominator = n_pos[n_pos > 0]
        scores[n_pos > 0] = n_relevant[n_pos > 0].astype(float) / denominator

        return scores

    def _score(self, y_true, y_pred_labels):
        return self._recall(y_true, y_pred_labels)
