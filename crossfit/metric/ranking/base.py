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

from crossfit.data.array.dispatch import crossarray
from crossfit.data.array.masked import MaskedArray
from crossfit.data.sparse.ranking import Rankings, SparseBinaryLabels, SparseLabels
from crossfit.metric.continuous.mean import Mean


class RankingMetric(Mean):
    def prepare(
        self,
    ):
        ...

    def score(self, y_true: SparseLabels, y_pred: Rankings, nan_handling="zerofill"):
        if not isinstance(y_true, SparseLabels):
            raise TypeError("y_true must be of type Labels")
        if not isinstance(y_pred, Rankings):
            raise TypeError("y_pred must be of type Rankings")

        y_pred_labels = y_true.get_labels_for(y_pred, self._k)
        scores = self._score(y_true, y_pred_labels)

        return self.nan_handling(scores, nan_handling)

    @classmethod
    def _bootstrap_ci(cls, scores, n_bootstrap_samples, confidence):
        if not isinstance(n_bootstrap_samples, int) or n_bootstrap_samples <= 1:
            raise ValueError("n_bootstrap_samples must be int > 1")
        elif not isinstance(confidence, float) or confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("Confidence must be float and 0 < confidence < 1")

        if len(scores):
            resamples = np.random.choice(scores, (len(scores), n_bootstrap_samples), replace=True)
            bootstrap_means = resamples.mean(axis=0)

            # Compute "percentile bootstrap"
            alpha_2 = (1 - confidence) / 2.0
            lower_ci = np.quantile(bootstrap_means, alpha_2)
            upper_ci = np.quantile(bootstrap_means, 1.0 - alpha_2)
            return (lower_ci, upper_ci)
        else:
            return (float("nan"), float("nan"))

    def mean(
        self,
        y_true,
        y_pred,
        nan_handling="propagate",
        conf_interval=False,
        n_bootstrap_samples=1000,
        confidence=0.95,
    ):
        scores = self.score(y_true, y_pred, nan_handling=nan_handling)

        if conf_interval:
            ci = self._bootstrap_ci(scores, n_bootstrap_samples, confidence)
        else:
            ci = None

        if len(scores):
            mean = scores.mean()
        else:
            mean = float("nan")

        if conf_interval:
            return {"score": mean, "conf_interval": ci}
        else:
            return mean

    def nan_handling(self, scores, handling="zerofill"):
        with crossarray:
            if handling == "drop":
                return scores[~np.isnan(scores)]
            elif handling == "zerofill":
                return np.nan_to_num(scores)
            elif handling == "propagate":
                return scores
            else:
                raise ValueError('nan_handling must be "propagate", "drop" or "zerofill"')

    def name(self):
        if self._k is None:
            k = ""
        else:
            k = f"@{self._k}"

        return self.__class__.__name__ + k


class BinaryRankingMetric(RankingMetric):
    def __init__(self, k):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Cutoff k needs to be integer > 0")
        self._k = k

    def score(self, y_true: SparseBinaryLabels, y_pred: MaskedArray, **kwargs):
        if not isinstance(y_true, SparseBinaryLabels):
            raise TypeError(
                f"y_true must be of type BinaryLabels but is of instance {type(y_true)}"
            )
        return super().score(y_true, y_pred, **kwargs)
