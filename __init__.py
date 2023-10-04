from crossfit import backend
from crossfit.backend.dask.cluster import setup_dask_cluster

from crossfit.calculate.aggregate import Aggregator
from crossfit.calculate.module import CrossModule

from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame

from crossfit.dataset.load import load_dataset
from crossfit.dataset.base import MultiDataset, IRDataset

from crossfit import op
from crossfit.op import *  # noqa

from crossfit import metric
from crossfit.metric import *  # noqa

from crossfit.report.beir.embed import embed
from crossfit.report.beir.report import beir_report


__all__ = [
    "Aggregator",
    "backend",
    "CrossModule",
    "CrossFrame",
    "crossarray",
    "convert_array",
    "FrameBackend",
    "load_dataset",
    "MultiDataset",
    "IRDataset",
    "op",
    "metric",
    "setup_dask_cluster",
    "embed",
    "beir_report",
]
