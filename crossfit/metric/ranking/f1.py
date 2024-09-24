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

from crossfit.metric.ranking.precision import Precision
from crossfit.metric.ranking.recall import Recall


class F1(Precision, Recall):
    def _score(self, y_true, y_pred_labels):
        recall = self._recall(y_true, y_pred_labels)
        precision = self._precision(y_true, y_pred_labels)

        product = 2 * recall * precision
        sm = recall + precision

        # return 0 for geometric mean if both are zero
        scores = np.zeros_like(product, dtype=float)
        valid = np.nan_to_num(product) > 0
        invalid = np.isnan(product)

        scores[valid] = product[valid] / sm[valid]
        scores[invalid] = np.nan

        return scores
