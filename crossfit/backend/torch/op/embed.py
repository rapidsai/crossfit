import gc

import cupy as cp
import cudf
import rmm
import torch

from crossfit.op.base import Op
from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.model import Model
from crossfit.backend.torch.loader import SortedSeqLoader, InMemoryLoader


class Embedder(Op):
    def __init__(
        self,
        model: Model,
        pre=None,
        cols=False,
        keep_cols=None,
        default_batch_size=1024,
        max_mem: str = "16GB",
        sorted_data_loader: bool = True,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model = model
        self.batch_size = default_batch_size
        self.max_mem = max_mem
        self.max_mem_gb = int(self.max_mem.split("GB")[0]) / 2.5
        self.sorted_data_loader = sorted_data_loader

    def setup(self):
        self.model.load_on_worker(self)

        # self.model = SentenceTransformer(self.model_name, device="cuda").to("cuda")
        # self.cfg = AutoConfig.from_pretrained("sentence-transformers/" + self.model_name)
        # self.memory_estimator = HFMemoryEstimator(self.max_mem_gb, self.cfg)

        # self.memory_estimator = NVMLMemoryEstimator(self.max_mem_gb, self.cfg)
        # if hasattr(self.memory_estimator, "fit"):
        #     self.memory_estimator.fit(self.model)

    @torch.no_grad()
    def call(self, data, partition_info=None):
        index = data.index
        if self.sorted_data_loader:
            loader = SortedSeqLoader(
                data[["input_ids", "attention_mask"]],
                self.model,
                progress_bar=self.create_progress_bar(len(data), partition_info),
                initial_batch_size=self.batch_size,
            )
        else:
            loader = InMemoryLoader(
                data[["input_ids", "attention_mask"]],
                batch_size=self.batch_size,
                progress_bar=self.create_progress_bar(len(data), partition_info),
                max_seq_len=self.model.max_seq_length(),
            )

        all_embeddings_ls = []
        for output in loader.map(self.model.get_model(self)):
            all_embeddings_ls.append(output["sentence_embedding"])

        out = cudf.DataFrame(index=index)
        embedding = cp.asarray(torch.vstack(all_embeddings_ls))
        _index = loader.sort_column(index.values) if self.sorted_data_loader else index
        out["embedding"] = create_list_series_from_2d_ar(embedding, _index)

        gc.collect()
        torch.cuda.empty_cache()

        return out

    def meta(self):
        return {"embedding": "float32"}
