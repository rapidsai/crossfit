import cupy as cp
import numpy as np
import pytest

import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS
from crossfit.op.vector_search import _per_dim_ddf

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")


@pytest.mark.multigpu
@pytest.mark.parametrize("dataset", DATASETS)
def test_beir_report_basic(dataset, model="all-MiniLM-L6-v2"):
    with cf.Distributed():
        embeds = cf.embed(dataset, model, tiny_sample=True)
        embeds = embeds.predictions.ddf().compute().to_pandas()

    assert set(embeds.columns) == set(
        ["corpus-index", "score", "query-id", "query-index"]
    )
    assert embeds["query-index"].nunique() == embeds["query-id"].nunique()
