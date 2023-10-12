import cupy as cp
import torch
from beir.retrieval.search.dense import util as utils

from crossfit.data.array.conversion import convert_array
from crossfit.op.vector_search import ExactSearchOp


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
        self.score_functions = {"cos_sim": utils.cos_sim, "dot": utils.dot_score}
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
