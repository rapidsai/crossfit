import pytest

cp = pytest.importorskip("cupy")

import random

import numpy as np

import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")
DATASETS.discard("webis-touche2020")
DATASETS.discard("trec-covid")


@pytest.mark.singlegpu
def test_embed_multi_gpu(
    model_name="all-MiniLM-L6-v2",
    rmm_pool_size="12GB",
    k=10,
):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    with cf.Distributed(rmm_pool_size=rmm_pool_size, n_workers=1):
        for dataset in DATASETS:
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


@pytest.mark.multigpu
def test_embed_multi_gpu(
    model_name="all-MiniLM-L6-v2",
    rmm_pool_size="24GB",
    k=10,
):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    with cf.Distributed(rmm_pool_size=rmm_pool_size, n_workers=2):
        for dataset in DATASETS:
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
