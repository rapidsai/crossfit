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

from crossfit.data.array.conversion import convert_array
from crossfit.data.array.masked import MaskedArray
from crossfit.metric.ranking.base import RankingMetric, SparseLabels


class DCG(RankingMetric):
    SCALERS = {"identity": lambda x: x, "power": lambda x: np.power(x, 2) - 1}
    LOGS = {"2": lambda x: np.log2(x), "e": lambda x: np.log(x)}

    def __init__(self, k=None, relevance_scaling="identity", log_base="2"):
        self._k = k
        if relevance_scaling not in self.SCALERS:
            raise ValueError("Relevance scaling must be 'identity' or 'power'.")
        if log_base not in self.LOGS:
            raise ValueError("Log base needs to be 'e' or '2'.")
        self._rel_scale = self.SCALERS[relevance_scaling]
        self._log_fct = self.LOGS[log_base]

    def _dcg(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, : self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)
        ranks = convert_array(ranks, type(y_pred_labels.data))

        scores = np.sum(self._rel_scale(labels) / self._log_fct(ranks + 1), axis=-1)
        scores[n_pos == 0] = np.nan
        return scores

    def _score(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        return self._dcg(y_true, y_pred_labels)


class NDCG(DCG):
    def _score(self, y_true: SparseLabels, y_pred_labels: MaskedArray):
        dcg = self._dcg(y_true, y_pred_labels)
        ideal_labels = y_true.get_labels_for(y_true.as_rankings(), self._k)
        idcg = self._dcg(y_true, ideal_labels)

        ndcg = dcg / idcg

        if idcg.shape[0] == 1 and ndcg.shape[0] > 1:
            idcg = np.ones_like(ndcg) * idcg

        ndcg[idcg == 0] = np.nan

        return dcg / idcg
