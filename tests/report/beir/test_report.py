import pytest

pytest.importorskip("cupy")
beir = pytest.importorskip("beir")

import numpy as np

import crossfit as cf
from crossfit.data.sparse.ranking import SparseNumericLabels, SparseRankings
from crossfit.metric.ranking import NDCG
from crossfit.report.beir.report import (
    create_csr_matrix,
    create_label_encoder,
    join_predictions,
)


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", ["nq", "hotpotqa", "fiqa"])
def test_beir_report(
    dataset,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    k=10,
    batch_size=128,
):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    report = cf.beir_report(
        dataset,
        model,
        vector_search=vector_search,
        overwrite=True,
        tiny_sample=True,
        batch_size=batch_size,
    )

    expected_columns = [
        f"{metric}@{k}"
        for metric in ["NDCG", "Recall", "Precision", "AP"]
        for k in [1, 3, 5, 10]
    ]
    expected_indices = [
        ("split", "test"),
        ("split", "train"),
        ("split", "val"),
    ]
    assert sorted(report.result_df.columns.tolist()) == sorted(expected_columns)
    assert ("split", "test") in report.result_df.index.values.tolist()
    for col in expected_columns:
        assert report.result_df.loc[("split", "test"), col].item() > 0.0


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", ["nq", "hotpotqa", "fiqa"])
def test_no_invalid_scores(
    dataset,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    k=10,
    batch_size=128,
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
    test = embeds.data.test.ddf()
    test["split"] = "test"
    df = join_predictions(test, embeds.predictions).compute()

    encoder = create_label_encoder(df, ["corpus-index-pred", "corpus-index-obs"])
    obs_csr = create_csr_matrix(df["corpus-index-obs"], df["score-obs"], encoder)
    pred_csr = create_csr_matrix(df["corpus-index-pred"], df["score-pred"], encoder)

    labels = SparseNumericLabels.from_matrix(obs_csr)
    rankings = SparseRankings.from_scores(pred_csr)

    ndcg = NDCG(5).score(labels, rankings)

    assert ndcg.min() >= 0
    assert ndcg.max() <= 1
    assert not np.isinf(ndcg).any()
