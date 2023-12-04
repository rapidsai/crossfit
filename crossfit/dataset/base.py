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

import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import dask_cudf

_SPLIT_ALIASES = {
    "val": ["validation", "valid", "dev"],
    "train": ["train", "training"],
    "test": ["test", "testing"],
}

_IR_ALIASES = {
    "query": ["query", "queries"],
    "item": ["item", "items", "corpus", "documents", "doc"],
}


@dataclass
class Dataset:
    path: str
    engine: str = "parquet"

    def ddf(self):
        return dask_cudf.read_parquet(self.path)


class FromDirMixin:
    @classmethod
    def from_dir(
        cls,
        dir: str,
        format: str = "parquet",
        dataset_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "MultiDataset":
        datasets = {}
        dataset_kwargs = dataset_kwargs or {}
        for name, aliases in cls.ALIASES.items():
            for alias in aliases:
                path = os.path.join(dir, alias)
                if os.path.isdir(path):
                    datasets[name] = os.path.join(path, f"*.{format}")
                    break
                path = os.path.join(dir, "qrels", alias)
                if os.path.isdir(path):
                    datasets[name] = os.path.join(path, f"*.{format}")
                    break

        if len(datasets) == 0:
            raise ValueError(f"Could not find any dataset in {dir}.")

        return cls(**datasets, **kwargs)


class MultiDataset(FromDirMixin):
    ALIASES = _SPLIT_ALIASES

    def __init__(
        self,
        train: Optional[Union[Dataset, str]] = None,
        val: Optional[Union[Dataset, str]] = None,
        test: Optional[Union[Dataset, str]] = None,
    ):
        self.train: Optional[Dataset] = Dataset(train) if isinstance(train, str) else train
        self.val: Optional[Dataset] = Dataset(val) if isinstance(val, str) else val
        self.test: Optional[Dataset] = Dataset(test) if isinstance(test, str) else test


class IRDataset(MultiDataset):
    ALIASES = {**_SPLIT_ALIASES, **_IR_ALIASES}

    def __init__(
        self,
        train: Optional[Union[Dataset, str]] = None,
        val: Optional[Union[Dataset, str]] = None,
        test: Optional[Union[Dataset, str]] = None,
        query: Optional[Union[Dataset, str]] = None,
        item: Optional[Union[Dataset, str]] = None,
    ):
        self.train: Optional[Dataset] = Dataset(train) if isinstance(train, str) else train
        self.val: Optional[Dataset] = Dataset(val) if isinstance(val, str) else val
        self.test: Optional[Dataset] = Dataset(test) if isinstance(test, str) else test
        self.query: Optional[Dataset] = Dataset(query) if isinstance(query, str) else query
        self.item: Optional[Dataset] = Dataset(item) if isinstance(item, str) else item


class EmbeddingDatataset(FromDirMixin):
    ALIASES = {"predictions": ["predictions", "topk"], **_IR_ALIASES}

    def __init__(
        self,
        query: Union[Dataset, str],
        item: Union[Dataset, str],
        predictions: Optional[Union[Dataset, str]] = None,
        data: Optional[IRDataset] = None,
    ):
        self.query: Dataset = Dataset(query) if isinstance(query, str) else query
        self.item: Dataset = Dataset(item) if isinstance(item, str) else item
        self.predictions: Optional[Dataset] = (
            Dataset(predictions) if isinstance(predictions, str) else predictions
        )
        self.data = data
