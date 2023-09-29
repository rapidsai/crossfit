from typing import Optional, Union
import os

import dask_cudf
from dataclasses import dataclass
from cuml.dask.neighbors import NearestNeighbors
from crossfit.backend.cudf.series import create_list_series_from_2d_ar


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
        format="parquet",
        dataset_kwargs=None,
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


class IRData(Dataset):
    def join_predictions(self, predictions):
        if hasattr(predictions, "ddf"):
            predictions = predictions.ddf()

        ddf = self.ddf()

        observed = (
            ddf[["query-index", "corpus-index", "score"]]
            .groupby("query-index")
            .agg(list, split_out=ddf.npartitions, shuffle=True)
        )

        predictions = predictions.set_index("query-index")
        merged = predictions.merge(
            observed, left_index=True, right_index=True, how="left", suffixes=("-pred", "-obs")
        )

        return merged.reset_index()


class IRDataset(MultiDataset):
    ALIASES = {**_SPLIT_ALIASES, **_IR_ALIASES}

    def __init__(
        self,
        train: Optional[Union[IRData, str]] = None,
        val: Optional[Union[IRData, str]] = None,
        test: Optional[Union[IRData, str]] = None,
        query: Optional[Union[Dataset, str]] = None,
        item: Optional[Union[Dataset, str]] = None,
    ):
        self.train: Optional[IRData] = IRData(train) if isinstance(train, str) else train
        self.val: Optional[IRData] = IRData(val) if isinstance(val, str) else val
        self.test: Optional[IRData] = IRData(test) if isinstance(test, str) else test
        self.query: Optional[Dataset] = Dataset(query) if isinstance(query, str) else query
        self.item: Optional[Dataset] = Dataset(item) if isinstance(item, str) else item


class EmbeddingData(Dataset):
    def per_dim_ddf(self, data=None) -> dask_cudf.DataFrame:
        import cudf

        _ddf = self.ddf() if data is None else data
        dim = len(_ddf.head()["embedding"].iloc[0])

        def to_map(part, dim):
            df = cudf.DataFrame(
                part["embedding"].list.leaves.values.reshape(-1, dim).astype("float32")
            )

            return df

        meta = {i: "float32" for i in range(int(dim))}
        output = _ddf.map_partitions(to_map, dim=dim, meta=meta)

        output.index = _ddf["index"]

        return output


class EmbeddingDatataset(FromDirMixin):
    ALIASES = {"predictions": ["predictions", "topk"], **_IR_ALIASES}

    def __init__(
        self,
        query: Union[EmbeddingData, str],
        item: Union[EmbeddingData, str],
        predictions: Optional[Union[Dataset, str]] = None,
        data: Optional[IRDataset] = None,
    ):
        self.query: EmbeddingData = EmbeddingData(query) if isinstance(query, str) else query
        self.item: EmbeddingData = EmbeddingData(item) if isinstance(item, str) else item
        self.predictions: Optional[Dataset] = (
            Dataset(predictions) if isinstance(predictions, str) else predictions
        )
        self.data = data
