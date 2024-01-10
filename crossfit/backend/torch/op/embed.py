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

from crossfit.backend.torch.loader import DEFAULT_BATCH_SIZE
from crossfit.backend.torch.model import Model
from crossfit.backend.torch.op.base import Predictor


class Embedder(Predictor):
    def __init__(
        self,
        model: Model,
        pre=None,
        cols=False,
        keep_cols=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_mem: str = "16GB",
        sorted_data_loader: bool = True,
        model_output_col: str = "sentence_embedding",
        pred_output_col: str = "embedding",
    ):
        super().__init__(
            model,
            pre=pre,
            cols=cols,
            keep_cols=keep_cols,
            batch_size=batch_size,
            max_mem=max_mem,
            sorted_data_loader=sorted_data_loader,
            model_output_col=model_output_col,
            pred_output_col=pred_output_col,
        )
