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

import cupy as cp
import torch

from crossfit.data.array.conversion import convert_array
from crossfit.op.vector_search import ExactSearchOp
from crossfit.utils import math_utils


class TorchExactSearch(ExactSearchOp):
    def __init__(
        self,
        k: int,
        pre=None,
        embedding_col="embedding",
        metric="cos_sim",
        keep_cols=None,
    ):
        super().__init__(pre=pre, keep_cols=keep_cols)
        self.k = k
        self.metric = metric
        self.embedding_col = embedding_col
        self.normalize = False
        self.score_functions = {"cos_sim": math_utils.cos_sim, "dot": math_utils.dot_score}
        self.score_function_desc = {"cos_sim": "Cosine Similarity", "dot": "Dot Product"}

    def search_tensors(self, queries, corpus):
        queries = convert_array(queries, torch.Tensor)
        corpus = convert_array(corpus, torch.Tensor)

        score_function = self.score_functions[self.metric]
        sim_scores = score_function(queries, corpus)
        sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(
            sim_scores, k=self.k, dim=1, largest=True, sorted=True
        )

        results = convert_array(sim_scores_top_k_values, cp.ndarray)
        indices = convert_array(sim_scores_top_k_idx, cp.ndarray)

        return results, indices
