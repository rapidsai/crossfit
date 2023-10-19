import pytest

beir = pytest.importorskip("beir")

import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")
DATASETS.discard("climate-fever")

# below are datasets with numerical scores > 1.
# TODO: remove after enabling non-binary scores.
DATASETS.discard("msmarco")
DATASETS.discard("nfcorpus")
DATASETS.discard("webis-touche2020")
DATASETS.discard("dbpedia-entity")
DATASETS.discard("trec-covid")


@pytest.mark.singlegpu
def test_beir_report(model_name="all-MiniLM-L6-v2", mem_size=12, k=10):
    model = cf.SentenceTransformerModel(model_name, max_mem_gb=mem_size)
    vector_search = cf.TorchExactSearch(k=k)

    with cf.Distributed(rmm_pool_size=f"{mem_size}GB", n_workers=1):
        for dataset in DATASETS:
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


#@pytest.mark.multigpu
#def test_beir_report(model_name="all-MiniLM-L6-v2", mem_size=12, k=10):
#    model = cf.SentenceTransformerModel(model_name, max_mem_gb=mem_size)
#    vector_search = cf.TorchExactSearch(k=k)
#
#    with cf.Distributed(rmm_pool_size=f"{mem_size}GB", n_workers=2):
#        for dataset in DATASETS:
#            report = cf.beir_report(
#                dataset,
#                model,
#                vector_search=vector_search,
#                overwrite=True,
#                tiny_sample=True,
#            )
#
#        expected_columns = [
#            f"{metric}@{k}"
#            for metric in ["NDCG", "Recall", "Precision"]
#            for k in [1, 3, 5, 10]
#        ]
#        expected_indices = [
#            ("split", "test"),
#            ("split", "train"),
#            ("split", "val"),
#        ]
#        assert sorted(report.result_df.columns.tolist()) == sorted(expected_columns)
#        assert ("split", "test") in report.result_df.index.values.tolist()
#        for col in expected_columns:
#            assert report.result_df.loc[("split", "test"), col].item() > 0.0
