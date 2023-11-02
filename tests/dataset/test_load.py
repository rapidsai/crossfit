import pytest

beir = pytest.importorskip("beir")

import crossfit as cf  # noqa: E402
from crossfit.dataset.beir.raw import BEIR_DATASETS  # noqa: E402

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")


@pytest.mark.singlegpu
@pytest.mark.parametrize("dataset", DATASETS)
def test_load_beir(dataset):
    data = cf.load_dataset(f"beir/{dataset}", overwrite=True, tiny_sample=True)

    for split in ["train", "val", "test"]:
        split_data = getattr(data, split)

        if split_data is None:
            continue

        split = split_data.ddf().compute()

        assert split["query-index"].nunique() == split["query-id"].nunique()
        assert split["query-id"].nunique() <= 100
