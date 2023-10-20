import pytest

beir = pytest.importorskip("beir")

import crossfit as cf


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", ["nq", "hotpotqa", "fiqa"])
def test_beir_report(dataset, model_name="all-MiniLM-L6-v2", k=10):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)
    report = cf.beir_report(
        dataset,
        model,
        vector_search=vector_search,
        overwrite=True,
        tiny_sample=True,
    )

    expected_columns = [
        f"{metric}@{k}"
        for metric in ["NDCG", "Recall", "Precision"]
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
