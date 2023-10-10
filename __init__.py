from crossfit import backend, metric, op
from crossfit.backend.dask.cluster import Distributed, Serial
from crossfit.backend.torch import SentenceTransformerModel, TorchExactSearch
from crossfit.calculate.aggregate import Aggregator
from crossfit.calculate.module import CrossModule
from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.dataset.base import IRDataset, MultiDataset
from crossfit.dataset.load import load_dataset
from crossfit.metric import *  # noqa
from crossfit.op import *  # noqa
from crossfit.report.beir.embed import embed
from crossfit.report.beir.report import beir_report

__all__ = [
    "Aggregator",
    "backend",
    "CrossModule",
    "CrossFrame",
    "crossarray",
    "convert_array",
    "Distributed",
    "FrameBackend",
    "load_dataset",
    "MultiDataset",
    "IRDataset",
    "op",
    "metric",
    "setup_dask_cluster",
    "Serial",
    "embed",
    "beir_report",
    "TorchExactSearch",
    "SentenceTransformerModel",
]
