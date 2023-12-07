# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

beir = pytest.importorskip("beir")

import crossfit as cf  # noqa: E402
from crossfit.dataset.beir.raw import BEIR_DATASETS  # noqa: E402

DATASETS = set(BEIR_DATASETS.keys())
DATASETS.discard("cqadupstack")
DATASETS.discard("germanquad")
DATASETS.discard("trec-covid")


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
