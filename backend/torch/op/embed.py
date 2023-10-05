import gc

import cupy as cp
import cudf
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from crossfit.op.base import Op
from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.hf.memory import HFMemoryEstimator
from crossfit.backend.torch.loader import SortedSeqLoader


class Embedder(Op):
    def __init__(
        self,
        model_name: str,
        pre=None,
        cols=False,
        default_batch_size=1024,
        max_mem: str = "16GB",
    ):
        super().__init__(pre, cols)
        self.model_name = model_name
        self.batch_size = default_batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2

    def setup(self):
        self.model = SentenceTransformer(self.model_name, device="cuda").to("cuda")
        self.cfg = AutoConfig.from_pretrained("sentence-transformers/" + self.model_name)
        self.memory_estimator = HFMemoryEstimator(self.max_mem_gb, self.cfg)

    @torch.no_grad()
    def call(self, data, partition_info=None):
        loader = SortedSeqLoader(
            data[["input_ids", "attention_mask"]],
            self.memory_estimator,
            progress_bar=self.create_progress_bar(len(data), partition_info),
        )

        data = loader.sort_df(data)
        all_embeddings_ls = []
        for output in loader.map(self.model):
            all_embeddings_ls.append(output["sentence_embedding"])

        out = cudf.DataFrame()
        out.index = data.index
        all_embeddings = torch.vstack(all_embeddings_ls)
        embedding = cp.asarray(all_embeddings)
        out["embedding"] = create_list_series_from_2d_ar(embedding, out.index)

        gc.collect()

        return out

    def meta(self):
        return {"embedding": "float32"}
