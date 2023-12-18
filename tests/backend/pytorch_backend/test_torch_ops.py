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

import crossfit as cf


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
