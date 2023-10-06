import gc

import cupy as cp
import cudf
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from crossfit.op.base import Op
from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.hf.memory import HFMemoryEstimator
from crossfit.backend.torch.loader import SortedSeqLoader, InMemoryLoader


class Embedder(Op):
    def __init__(
        self,
        model_name: str,
        pre=None,
        cols=False,
        keep_cols=None,
        default_batch_size=1024,
        max_mem: str = "16GB",
        sorted_data_loader: bool = False,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model_name = model_name
        self.batch_size = default_batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2
        self.sorted_data_loader = sorted_data_loader

    def setup(self):
        self.model = SentenceTransformer(self.model_name, device="cuda").to("cuda")
        self.cfg = AutoConfig.from_pretrained("sentence-transformers/" + self.model_name)
        self.memory_estimator = HFMemoryEstimator(self.max_mem_gb, self.cfg)

    @torch.no_grad()
    def call(self, data, partition_info=None):
        index = data.index
        if self.sorted_data_loader:
            loader = SortedSeqLoader(
                data[["input_ids", "attention_mask"]],
                self.memory_estimator,
                progress_bar=self.create_progress_bar(len(data), partition_info),
            )
        else:
            loader = InMemoryLoader(
                data[["input_ids", "attention_mask"]],
                batch_size=self.batch_size,
                progress_bar=self.create_progress_bar(len(data), partition_info),
            )

        all_embeddings_ls = []
        for output in loader.map(self.model):
            all_embeddings_ls.append(output["sentence_embedding"])

        out = cudf.DataFrame(index=index)
        embedding = cp.asarray(torch.vstack(all_embeddings_ls))
        _index = loader.sort_column(index.values) if self.sorted_data_loader else index
        out["embedding"] = create_list_series_from_2d_ar(embedding, _index)

        gc.collect()

        return out

    def meta(self):
        return {"embedding": "float32"}
