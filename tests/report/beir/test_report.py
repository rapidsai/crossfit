import pytest


import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS


DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")
DATASETS.discard("climate-fever")


@pytest.mark.multigpu
@pytest.mark.parametrize("dataset", DATASETS)
def test_beir_report(dataset, model_name="all-MiniLM-L6-v2", k=10):
    model = cf.SentenceTransformerModel(model_name)
    vector_search = cf.TorchExactSearch(k=k)

    with cf.Distributed():
        report = cf.beir_report(
            dataset, model, vector_search=dense_search, overwrite=True, tiny_sample=True
        )

    expected_columns = [
        f"{metric}@{k}" for metric in ["NDCG", "Recall", "Precision"] for k in [1, 3, 5, 10]
    ]
    expected_indices = [
        ("split", "test"),
        ("split", "train"),
        ("split", "val"),
    ]
    assert sorted(report.result_df.columns.tolist()) == sorted(expected_columns)
    assert sorted(report.result_df.index.values.tolist()) == sorted(expected_indices)
    for idx, col in zip(expected_indices, expected_columns):
        assert report.result_df.loc[idx, col].item() > 0.0
