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
Model = pytest.importorskip("crossfit.backend.torch.model").Model
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

        assert hasattr(mock_worker, f"torch_model_{id(model)}")
        assert hasattr(mock_worker, f"cfg_{id(model)}")

        model.unload_from_worker(mock_worker)

        assert not hasattr(mock_worker, f"torch_model_{id(model)}")
        assert not hasattr(mock_worker, f"cfg_{id(model)}")


class DummyModelWithDictOutput(torch.nn.Module):
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
        super().__init__(model_name, model_output_type={"a": "numeric", "b": "numeric"})

    def load_model(self, device="cuda"):
        return DummyModelWithDictOutput().to(device)


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


class DummyModelForMetaTest(Model):
    def __init__(self, model_output_type):
        super().__init__("dummy_model_path")
        self.model_output_type = model_output_type


class TestPredictorMeta:
    def setup_method(self):
        self.model_string = DummyModelForMetaTest(model_output_type="string")
        self.model_numeric = DummyModelForMetaTest(model_output_type="numeric")
        self.model_dict = DummyModelForMetaTest(model_output_type={"a": "string", "b": "numeric"})

    def test_meta_single_output_column_string(self):
        predictor = cf.op.Predictor(
            model=self.model_string, model_output_cols=["a"], pred_output_col="pred_a"
        )
        expected_output = {"pred_a": "object"}
        assert predictor.meta() == expected_output

    def test_meta_single_output_column_numeric(self):
        predictor = cf.op.Predictor(
            model=self.model_numeric, model_output_cols=["a"], pred_output_col="pred_a"
        )
        expected_output = {"pred_a": "float32"}
        assert predictor.meta() == expected_output

    def test_meta_multiple_output_columns(self):
        predictor = cf.op.Predictor(model=self.model_dict, model_output_cols=["a", "b"])
        expected_output = {"a": "object", "b": "float32"}
        assert predictor.meta() == expected_output

    def test_meta_no_model_output_cols_specified_string(self):
        predictor = cf.op.Predictor(
            model=self.model_string, model_output_cols=None, pred_output_col="preds"
        )
        expected_output = {"preds": "object"}
        assert predictor.meta() == expected_output

    def test_meta_no_model_output_cols_specified_numeric(self):
        predictor = cf.op.Predictor(
            model=self.model_numeric, model_output_cols=None, pred_output_col="preds"
        )
        expected_output = {"preds": "float32"}
        assert predictor.meta() == expected_output

    def test_meta_invalid_model_output_type(self):
        with pytest.raises(
            ValueError,
            match=(
                "model_output_type must be a dictionary when multiple "
                "model_output_cols are specified"
            ),
        ):
            predictor = cf.op.Predictor(model=self.model_string, model_output_cols=["a", "b"])
            predictor.meta()


class DummyHFModel_WithOutputValue(HFModel):
    def __init__(self, model_name, output_value):
        self.model_name = model_name
        self.output_value = output_value
        super().__init__(model_name)

    def load_model(self, device="cuda"):
        class DummyModel(torch.nn.Module):
            def __init__(self, output_value):
                super().__init__()
                self.output_value = output_value

            def forward(self, batch):
                output_size = len(batch["input_ids"])
                return torch.ones(output_size, device="cuda") * self.output_value

        return DummyModel(output_value=self.output_value).to(device)


def test_loading_multiple_models():
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0")
    client = Client(cluster)

    ddf = dask_cudf.from_cudf(cudf.DataFrame({"text": ["apple"] * 6}), npartitions=1)
    model_1 = DummyHFModel_WithOutputValue("microsoft/deberta-v3-base", 1)
    model_2 = DummyHFModel_WithOutputValue("microsoft/deberta-v3-base", 2)

    pipe_1 = cf.op.Sequential(
        cf.op.Tokenizer(model_1, cols=["text"], tokenizer_type="sentencepiece"),
        cf.op.Predictor(model_1, sorted_data_loader=False, batch_size=2, pred_output_col="pred_1"),
        keep_cols=list(ddf.columns),
    )
    output_1_ddf = pipe_1(ddf)
    pipe_2 = cf.op.Sequential(
        cf.op.Tokenizer(model_2, cols=["text"], tokenizer_type="sentencepiece"),
        cf.op.Predictor(model_2, sorted_data_loader=False, batch_size=2, pred_output_col="pred_2"),
        keep_cols=list(output_1_ddf.columns),
    )
    output_2_ddf = pipe_2(output_1_ddf)
    output_2_df = output_2_ddf.to_backend("pandas").compute()
    assert output_2_df["pred_1"].values.tolist() == [1, 1, 1, 1, 1, 1]
    assert output_2_df["pred_2"].values.tolist() == [2, 2, 2, 2, 2, 2]

    del client
    del cluster
