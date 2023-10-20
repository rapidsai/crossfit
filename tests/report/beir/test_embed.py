import pytest

cp = pytest.importorskip("cupy")

import random

import numpy as np

import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")
DATASETS.discard("trec-covid")
DATASETS.discard("webis-touche2020")


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", DATASETS)
def test_embed_multi_gpu(
    dataset,
    model_name="all-MiniLM-L6-v2",
    k=10,
):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    embeds = cf.embed(
        dataset,
        model,
        vector_search=vector_search,
        overwrite=True,
        tiny_sample=True,
    )
    embeds = embeds.predictions.ddf().compute().to_pandas()

    assert set(embeds.columns) == set(
        ["corpus-index", "score", "query-id", "query-index"]
    )
    assert embeds["query-index"].nunique() == embeds["query-id"].nunique()
