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


class LazyLoader:
    def __init__(self, name):
        self._name = name
        self._module = None
        self._error = None

    def _load(self):
        if self._module is None and self._error is None:
            try:
                parts = self._name.split(".")
                module_name = ".".join(parts[:-1])
                attribute_name = parts[-1]
                module = __import__(module_name, fromlist=[attribute_name])
                self._module = getattr(module, attribute_name)
            except ImportError as e:
                self._error = e
            except AttributeError as e:
                self._error = AttributeError(
                    f"Module '{module_name}' has no attribute '{attribute_name}'"
                )

    def __getattr__(self, item):
        self._load()
        if self._error is not None:
            raise ImportError(f"Failed to import {self._name}: {self._error}")
        return getattr(self._module, item)

    def __call__(self, *args, **kwargs):
        self._load()
        if self._error is not None:
            raise ImportError(f"Failed to import {self._name}: {self._error}")
        return self._module(*args, **kwargs)


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

# Using the lazy import function
HFModel = LazyLoader("crossfit.backend.torch.HFModel")
SentenceTransformerModel = LazyLoader("crossfit.backend.torch.SentenceTransformerModel")
TorchExactSearch = LazyLoader("crossfit.backend.torch.TorchExactSearch")
IRDataset = LazyLoader("crossfit.dataset.base.IRDataset")
MultiDataset = LazyLoader("crossfit.dataset.base.MultiDataset")
load_dataset = LazyLoader("crossfit.dataset.load.load_dataset")
embed = LazyLoader("crossfit.report.beir.embed.embed")
beir_report = LazyLoader("crossfit.report.beir.report.beir_report")
utils = LazyLoader("crossfit.utils")

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
    ]
)
