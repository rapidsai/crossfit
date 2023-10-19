import pytest

beir = pytest.importorskip("beir")

import crossfit as cf
from crossfit.dataset.beir.raw import BEIR_DATASETS

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")


@pytest.mark.multigpu
@pytest.mark.parametrize("dataset", DATASETS)
def test_load_beir(dataset):
    data = cf.load_dataset(f"beir/{dataset}", tiny_sample=True)

    for split in ["train", "val", "test"]:
        split_data = getattr(data, split)

        if split_data is None:
            continue

        split = split_data.ddf().compute()

        assert split["query-index"].nunique() == split["query-id"].nunique()
        assert split["query-id"].nunique() <= 100


@pytest.mark.multigpu
@pytest.mark.parametrize(
    "dataset,dtype",
    [
        ("nfcorpus", "query"),
        ("nfcorpus", "item"),
    ],
)
def test_load_dataset_with_dask(
    tmp_path,
    dataset,
    dtype,
    model_name="all-MiniLM-L6-v2",
    npartitions=2,
):
    with cf.Distributed():
        ir_dataset = cf.load_dataset(f"beir/{dataset}")
        df = getattr(ir_dataset, f"{dtype}_ddf")

        model_name = "all-MiniLM-L6-v2"
        pipe = cf.Sequential(
            cf.Tokenizer(model_name, cols=["text"]),
            cf.Embedder(model_name),
            repartition=npartitions,
        )

        pipe(df).to_parquet(tmp_path, write_index=False)
