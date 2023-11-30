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

from crossfit.data.sparse.ranking import (
    Rankings,
    SparseBinaryLabels,
    SparseLabels,
    SparseNumericLabels,
    SparseRankings,
)
from crossfit.metric.ranking.f1 import F1
from crossfit.metric.ranking.hitrate import HitRate
from crossfit.metric.ranking.ndcg import DCG, NDCG
from crossfit.metric.ranking.precision import AP, Precision
from crossfit.metric.ranking.rank import FirstRelevantRank, MeanRanks, ReciprocalRank
from crossfit.metric.ranking.recall import Recall

__all__ = [
    "AP",
    "F1",
    "FirstRelevantRank",
    "DCG",
    "HitRate",
    "MeanRanks",
    "NDCG",
    "Precision",
    "ReciprocalRank",
    "Recall",
    "SparseLabels",
    "SparseBinaryLabels",
    "SparseNumericLabels",
    "SparseRankings",
    "Rankings",
]
