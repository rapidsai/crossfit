from beir.retrieval.search.dense import util as utils
import cudf
import cupy as cp
import torch

from crossfit.data.array.conversion import convert_array
from crossfit.op.vector_search import ExactSearchOp, _get_embedding_cupy
from crossfit.backend.cudf.series import create_list_series_from_2d_ar


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
        self.desc = True
        self.embedding_col = embedding_col
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

    def call(self, queries, items):
        query_emb = _get_embedding_cupy(queries, self.embedding_col, normalize=True)
        item_emb = _get_embedding_cupy(items, self.embedding_col, normalize=True)

        results, indices = self.search_tensors(query_emb, item_emb)

        df = cudf.DataFrame(index=queries.index)
        df["query-id"] = queries["_id"]
        df["query-index"] = queries["index"]
        df["corpus-index"] = create_list_series_from_2d_ar(items["index"].values[indices], df.index)
        df["score"] = create_list_series_from_2d_ar(results, df.index)

        return df
