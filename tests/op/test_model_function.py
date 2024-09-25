# Copyright 2024 NVIDIA CORPORATION
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
from unittest.mock import patch

cp = pytest.importorskip("cupy")
cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")
dd = pytest.importorskip("dask.dataframe")
pd = pytest.importorskip("pandas")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

import crossfit as cf  # noqa: E402


cf_loader = pytest.importorskip("crossfit.backend.torch.loader")


@pytest.mark.parametrize("trust_remote_code", ["y"])
def test_model_output_int(trust_remote_code, model_name="microsoft/deberta-v3-base"):
    with patch("builtins.input", return_value=trust_remote_code):
        tokens_data = cudf.DataFrame({"input_ids": [[11, 12, 13], [14, 15, 16], [17, 18, 19]]})
        index = tokens_data.index.copy()
        model = cf.HFModel(model_name)
        data = [[4], [7], [10]]
        all_outputs_ls = torch.tensor(data)
        loader = cf_loader.SortedSeqLoader(
            tokens_data,
            model,
        )
        pred_output_col = "translation"
        out = model.get_model_output(all_outputs_ls, index, loader, pred_output_col)
        assert isinstance(out, cudf.DataFrame)
        assert isinstance(out["translation"][0][0], int)
        assert (
            out["translation"][0] == data[2]
            and out["translation"][1] == data[1]
            and out["translation"][2] == data[0]
        )


@pytest.mark.parametrize("trust_remote_code", ["y"])
def test_model_output_str(trust_remote_code, model_name="microsoft/deberta-v3-base"):
    with patch("builtins.input", return_value=trust_remote_code):
        tokens_data = cudf.DataFrame(
            {"input_ids": [[18264, 7728, 8], [123, 99, 2258], [3115, 125, 123]]}
        )
        index = tokens_data.index.copy()
        model = cf.HFModel(model_name, model_output_type="string")
        data = [["▁हमारे▁परीक्षण▁डेटा"], ["▁पर▁हमारे▁दो"], ["▁दूरी▁कार्यों▁की"]]
        all_outputs_ls = data
        loader = cf_loader.SortedSeqLoader(
            tokens_data,
            model,
        )
        pred_output_col = "translation"
        out = model.get_model_output(all_outputs_ls, index, loader, pred_output_col)
        assert isinstance(out, cudf.DataFrame)
        assert isinstance(out["translation"][0][0], str)
        assert (
            out["translation"][0] == data[2][0]
            and out["translation"][1] == data[1][0]
            and out["translation"][2] == data[0][0]
        )
