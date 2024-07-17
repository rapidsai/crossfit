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

cp = pytest.importorskip("cupy")
sentece_transformers = pytest.importorskip("sentence_transformers")

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
