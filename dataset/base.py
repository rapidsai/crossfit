from typing import Optional
import os

import dask_cudf
from dataclasses import dataclass

_SPLIT_ALIASES = {
    "val_path": ["validation", "valid", "dev"],
    "train_path": ["train", "training"],
    "test_path": ["test", "testing"],
}

_IR_ALIASES = {
    "query_path": ["query", "queries"],
    "item_path": ["item", "items", "corpus", "documents", "doc"],
}


@dataclass
class MultiDataset:
    ALIASES = _SPLIT_ALIASES

    train_path: Optional[str] = None
    test_path: Optional[str] = None
    val_path: Optional[str] = None

    @classmethod
    def from_dir(
        cls,
        dir: str,
        format="parquet",
        dataset_kwargs=None,
    ) -> "MultiDataset":
        datasets = {}
        dataset_kwargs = dataset_kwargs or {}
        for name, aliases in cls.ALIASES.items():
            for alias in aliases:
                path = os.path.join(dir, alias)
                if os.path.isdir(path):
                    datasets[name] = os.path.join(path, f"*.{format}")
                    break
                path = os.path.join(dir, "qrels", name)
                if os.path.isdir(path):
                    datasets[name] = os.path.join(path, f"*.{format}")
                    break

        if len(datasets) == 0:
            raise ValueError(f"Could not find any dataset in {dir}.")

        return cls(**datasets)

    @property
    def train_ddf(self):
        if not self.train_path:
            raise ValueError("No train dataset found.")

        return dask_cudf.read_parquet(self.train_path)

    @property
    def val_ddf(self):
        if not self.val_path:
            raise ValueError("No val dataset found.")

        return dask_cudf.read_parquet(self.val_path)

    @property
    def test_ddf(self):
        if not self.test_path:
            raise ValueError("No test dataset found.")

        return dask_cudf.read_parquet(self.test_path)


@dataclass
class IRDataset(MultiDataset):
    ALIASES = {**_SPLIT_ALIASES, **_IR_ALIASES}

    query_path: Optional[str] = None
    item_path: Optional[str] = None

    @property
    def query_ddf(self):
        if not self.query_path:
            raise ValueError("No query dataset found.")
        return dask_cudf.read_parquet(self.query_path)

    @property
    def item_ddf(self):
        if not self.item_path:
            raise ValueError("No item dataset found.")
        return dask_cudf.read_parquet(self.item_path)
