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


from typing import Any, List

import cudf
import cupy as cp

from crossfit.backend.cudf.series import (
    create_list_series_from_1d_or_2d_ar,
    create_nested_list_series_from_3d_ar,
)
from crossfit.utils.torch_utils import cleanup_torch_cache, concat_and_pad_tensors


class Model:
    def __init__(self, path_or_name: str, max_mem_gb: int = 16, model_output_type: str = "numeric"):
        self.path_or_name = path_or_name
        self.max_mem_gb = max_mem_gb
        if model_output_type in ["numeric", "string"]:
            self.model_output_type = model_output_type
        else:
            raise ValueError(
                "Invalid model output type provided. Allowed values are : 'string' or 'numeric'."
            )

    def load_model(self, device="cuda"):
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def load_on_worker(self, worker):
        raise NotImplementedError()

    def unload_from_worker(self, worker):
        raise NotImplementedError()

    def call_on_worker(self, worker, *args, **kwargs):
        return worker.torch_model(*args, **kwargs)

    def get_model(self, worker):
        if not hasattr(worker, "torch_model"):
            self.load_on_worker(worker)
        return worker.torch_model

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        raise NotImplementedError()

    def max_seq_length(self) -> int:
        raise NotImplementedError()

    def get_model_output(self, all_outputs_ls, index, loader, pred_output_col) -> cudf.DataFrame:
        # importing here to avoid cyclic import error
        from crossfit.backend.torch.loader import SortedSeqLoader

        out_df = cudf.DataFrame(index=index)
        _index = loader.sort_column(index.values) if type(loader) is SortedSeqLoader else index
        if isinstance(all_outputs_ls[0], dict):
            for pred_name in all_outputs_ls[0].keys():
                _add_column_to_df(
                    out_df,
                    [o[pred_name] for o in all_outputs_ls],
                    _index,
                    loader,
                    pred_name,
                    self.model_output_type,
                )
        else:
            _add_column_to_df(
                out_df, all_outputs_ls, _index, loader, pred_output_col, self.model_output_type
            )
        cleanup_torch_cache()
        return out_df


def _add_column_to_df(
    df: cudf.DataFrame,
    all_outputs_ls: List[Any],
    _index: Any,
    loader: Any,
    pred_output_col: str,
    model_output_type: str,
) -> None:
    if model_output_type == "string":
        _add_string_column(df, pred_output_col, all_outputs_ls)
    else:
        _add_numeric_column(df, all_outputs_ls, _index, loader, pred_output_col)


def _add_string_column(
    df: cudf.DataFrame, pred_output_col: str, all_outputs_ls: List[List[str]]
) -> None:
    df[pred_output_col] = [o for output in all_outputs_ls for o in output]


def _add_numeric_column(
    df: cudf.DataFrame, all_outputs_ls: List[Any], _index: Any, loader: Any, pred_output_col: str
) -> None:
    outputs = cp.asarray(
        concat_and_pad_tensors(
            all_outputs_ls,
            pad_token_id=getattr(loader, "pad_token_id", None),
            padding_side=getattr(loader, "padding_side", None),
        )
    )
    del all_outputs_ls
    del loader
    cleanup_torch_cache()
    if len(outputs.shape) == 1:
        df[pred_output_col] = cudf.Series(outputs, index=_index)
    elif len(outputs.shape) == 2:
        df[pred_output_col] = create_list_series_from_1d_or_2d_ar(outputs, _index)
    elif len(outputs.shape) == 3:
        df[pred_output_col] = create_nested_list_series_from_3d_ar(outputs, _index)
    else:
        raise RuntimeError(f"Unexpected output shape: {outputs.shape}")
