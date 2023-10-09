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

        model = self.model.get_model(self)
        all_embeddings_ls = []
        # for output in loader.map(_safe_apply_model(model)):
        #     all_embeddings_ls.append(output["sentence_embedding"])

        for batch in loader:
            outputs = model(batch)
            all_embeddings_ls.append(outputs["sentence_embedding"])

        out = cudf.DataFrame(index=index)
        embedding = cp.asarray(torch.vstack(all_embeddings_ls))
        _index = loader.sort_column(index.values) if self.sorted_data_loader else index
        out["embedding"] = create_list_series_from_2d_ar(embedding, _index)

        gc.collect()
        torch.cuda.empty_cache()

        return out

    def meta(self):
        return {"embedding": "float32"}


def _safe_apply_model(model):
    def _inner(batch):
        try:
            to_call = model
            outputs = to_call(batch)

            return outputs
        except Exception as e:
            if (
                "out of memory" in str(e)
                or "CUDA error at" in str(e)
                or isinstance(e, (SystemError, MemoryError))
            ):
                batch_size = list(batch.values())[0].shape[0]
                to_call = model

                device = next(model.parameters()).device
                cpu_batch = {key: val.cpu() for key, val in batch.items()}
                cpu_to_call = to_call.cpu()

                del batch
                del to_call

                gc.collect()
                torch.cuda.empty_cache()
                rmm.reinitialize(pool_allocator=False)  # Resetting RMM

                to_call = cpu_to_call.to(device)
                batch = {key: val.to(device) for key, val in cpu_batch.items()}

                # Split the batch in half and try again
                first_half = {key: val[: batch_size // 2] for key, val in batch.items()}
                second_half = {key: val[batch_size // 2 :] for key, val in batch.items()}

                first = model(first_half)
                second = model(second_half)

                # Stack outputs based on their type
                if isinstance(first, dict) and isinstance(second, dict):
                    return {key: torch.cat([first[key], second[key]], dim=0) for key in first}
                elif isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                    return torch.cat([first, second], dim=0)
                else:
                    raise TypeError("Inconsistent output type from the model.")
            else:
                raise e

    return _inner
