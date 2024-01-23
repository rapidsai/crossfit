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

# flake8: noqa

from crossfit import backend, metric, op
from crossfit.backend.dask.cluster import Distributed, Serial
from crossfit.calculate.aggregate import Aggregator
from crossfit.calculate.module import CrossModule
from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.metric import *
from crossfit.op import *

__all__ = [
    "Aggregator",
    "backend",
    "CrossModule",
    "CrossFrame",
    "crossarray",
    "convert_array",
    "Distributed",
    "FrameBackend",
    "op",
    "metric",
    "setup_dask_cluster",
    "Serial",
]


try:
    from crossfit.backend.torch import HFModel, HFGenerator, SentenceTransformerModel, TorchExactSearch
    from crossfit.dataset.base import IRDataset, MultiDataset
    from crossfit.dataset.load import load_dataset
    from crossfit.report.beir.embed import embed
    from crossfit.report.beir.report import beir_report

    __all__.extend(
        [
            "embed",
            "beir_report",
            "load_dataset",
            "TorchExactSearch",
            "SentenceTransformerModel",
            "HFModel",
            "MultiDataset",
            "IRDataset",
            "HFGenerator",
        ]
    )
except ImportError as e:
    pass
