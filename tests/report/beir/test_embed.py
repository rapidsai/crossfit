import pytest

cp = pytest.importorskip("cupy")

import crossfit as cf  # noqa: E402


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", ["fiqa", "hotpotqa", "nq"])
def test_embed_multi_gpu(
    dataset,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    k=10,
    batch_size=64,
):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    embeds = cf.embed(
        dataset,
        model,
        vector_search=vector_search,
        overwrite=True,
        tiny_sample=True,
        batch_size=batch_size,
    )
    embeds = embeds.predictions.ddf().compute().to_pandas()

    assert set(embeds.columns) == set(["corpus-index", "score", "query-id", "query-index"])
    assert embeds["query-index"].nunique() == embeds["query-id"].nunique()
