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

from enum import Enum
from typing import Any, Dict, List, Union

import cudf
import cupy as cp
import torch

from crossfit.backend.cudf.series import (
    create_list_series_from_1d_or_2d_ar,
    create_nested_list_series_from_3d_ar,
)
from crossfit.utils.torch_utils import cleanup_torch_cache, concat_and_pad_tensors


class ModelOutputType(Enum):
    NUMERIC = "numeric"
    STRING = "string"


class Model:
    def __init__(
        self,
        path_or_name: str,
        max_mem_gb: int = 16,
        model_output_type: Union[str, Dict[str, str]] = "numeric",
    ):
        """Initialize a Crossfit Pytorch Model Instance.

        Args:
            path_or_name (str): Path to the model file or the model name to load.
            max_mem_gb (int): Maximum memory in gigabytes to allocate for the model.Defaults to 16.
            model_output_type (str, dict): Specifies the type of model output. Can be either
                "numeric" or "string". If a dictionary is provided, it maps prediction names to
                their respective types. Defaults to "numeric".
        """
        self.path_or_name = path_or_name
        self.max_mem_gb = max_mem_gb
        self.model_output_type = _validate_model_output_type(model_output_type)
        self._model_id = f"torch_model_{id(self)}"

    def load_model(self, device="cuda"):
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def load_on_worker(self, worker):
        raise NotImplementedError()

    def unload_from_worker(self, worker):
        raise NotImplementedError()

    def call_on_worker(self, worker, *args, **kwargs):
        return getattr(worker, self._model_id)(*args, **kwargs)

    def get_model(self, worker):
        if not hasattr(worker, self._model_id):
            self.load_on_worker(worker)
        return getattr(worker, self._model_id)

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        raise NotImplementedError()

    def max_seq_length(self) -> int:
        raise NotImplementedError()

    def get_model_output(
        self,
        all_outputs_ls: List[Union[dict, torch.Tensor]],
        index: Union[cudf.Index],
        loader: Any,
        pred_output_col: str,
    ) -> cudf.DataFrame:
        # importing here to avoid cyclic import error
        from crossfit.backend.torch.loader import SortedSeqLoader

        out_df = cudf.DataFrame(index=index)
        _index = loader.sort_column(index.values) if type(loader) is SortedSeqLoader else index
        if isinstance(all_outputs_ls[0], dict):
            if not isinstance(self.model_output_type, dict):
                raise ValueError(
                    "model_output_type must be a dictionary when the model output is a dictionary"
                )
            for pred_name in all_outputs_ls[0].keys():
                if pred_name not in self.model_output_type:
                    raise ValueError(
                        f"Invalid prediction name '{pred_name}'.\n"
                        f"Allowed prediction names: {list(self.model_output_type.keys())}\n"
                        "Please provide a valid prediction name ands its datatype "
                        "in the model_output_type dictionary."
                    )
                model_output_type = self.model_output_type.get(pred_name, self.model_output_type)
                _add_column_to_df(
                    out_df,
                    [o[pred_name] for o in all_outputs_ls],
                    _index,
                    loader,
                    pred_name,
                    model_output_type,
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
    model_output_type: ModelOutputType,
) -> None:
    if model_output_type is ModelOutputType.STRING:
        _add_string_column(df, pred_output_col, all_outputs_ls)
    elif model_output_type is ModelOutputType.NUMERIC:
        _add_numeric_column(df, all_outputs_ls, _index, loader, pred_output_col)
    else:
        raise ValueError(f"Invalid model_output_type: {model_output_type}")


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


def _validate_model_output_type(
    model_output_type: Union[str, ModelOutputType, dict[str, Union[str, ModelOutputType]]]
) -> Union[ModelOutputType, dict[str, ModelOutputType]]:
    """Validate and convert model output type to proper enum format.

    Args:
        model_output_type: Either a string/enum value, or a dict of string/enum values

    Returns:
        ModelOutputType or dict: Validated and converted output type(s)

    Raises:
        ValueError: If invalid output type is provided
    """

    def _convert_single_type(value):
        if isinstance(value, str):
            try:
                return ModelOutputType(value)
            except ValueError:
                raise ValueError(
                    f"Invalid model_output_type: {value}. "
                    f"Allowed values are: {[e.value for e in ModelOutputType]}"
                )
        elif isinstance(value, ModelOutputType):
            return value
        else:
            raise ValueError(
                f"model_output_type must be string or ModelOutputType, got {type(value)}"
            )

    if isinstance(model_output_type, dict):
        return {key: _convert_single_type(value) for key, value in model_output_type.items()}
    else:
        return _convert_single_type(model_output_type)
