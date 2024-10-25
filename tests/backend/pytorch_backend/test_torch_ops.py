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
from unittest.mock import Mock

import pytest
from distributed import Client

import crossfit as cf

cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
HFModel = pytest.importorskip("crossfit.backend.torch").HFModel
LocalCUDACluster = pytest.importorskip("dask_cuda").LocalCUDACluster


class TestHFModel:
    @pytest.fixture
    def model(self):
        return cf.HFModel("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.fixture
    def mock_worker(self):
        return Mock()

    def test_unload_from_worker(self, model, mock_worker):
        model.load_on_worker(mock_worker)

        assert hasattr(mock_worker, "torch_model")
        assert hasattr(mock_worker, "cfg")

        model.unload_from_worker(mock_worker)

        assert not hasattr(mock_worker, "torch_model")
        assert not hasattr(mock_worker, "cfg")


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        output_size = len(batch["input_ids"])
        return {
            "a": torch.ones(output_size, device="cuda") * 1,
            "b": torch.ones(output_size, device="cuda") * 2,
        }


class DummyHFModel(HFModel):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self, device="cuda"):
        return DummyModel().to(device)


def test_hf_model():
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0")
    client = Client(cluster)

    model = DummyHFModel()
    pipe = cf.op.Sequential(
        cf.op.Tokenizer(model, cols=["text"], tokenizer_type="sentencepiece"),
        cf.op.Predictor(
            model, sorted_data_loader=False, batch_size=2, model_output_cols=["a", "b"]
        ),
    )
    ddf = dask_cudf.from_cudf(cudf.DataFrame({"text": ["apple"] * 6}), npartitions=1)
    outputs = pipe(ddf).compute()
    assert outputs.a.values.tolist() == [1, 1, 1, 1, 1, 1]
    assert outputs.b.values.tolist() == [2, 2, 2, 2, 2, 2]

    del client
    del cluster
