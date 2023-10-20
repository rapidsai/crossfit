from crossfit import backend, metric, op
from crossfit.backend.dask.cluster import Distributed, Serial
from crossfit.calculate.aggregate import Aggregator
from crossfit.calculate.module import CrossModule
from crossfit.data.array.conversion import convert_array
from crossfit.data.array.dispatch import crossarray
from crossfit.data.dataframe.core import FrameBackend
from crossfit.data.dataframe.dispatch import CrossFrame
from crossfit.metric import *  # noqa
from crossfit.op import *  # noqa

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
    from crossfit.backend.torch import (
        HFModel,
        SentenceTransformerModel,
        TorchExactSearch,
    )
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
        ]
    )
except ImportError as e:
    pass
