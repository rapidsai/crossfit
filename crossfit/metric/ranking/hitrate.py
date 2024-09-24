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

from crossfit.metric.ranking.recall import Recall


class HitRate(Recall):
    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        scores = self._recall(y_true, y_pred_labels)
        scores[n_pos != 1] = np.nan  # Not defined for no or multiple positives
        return scores
