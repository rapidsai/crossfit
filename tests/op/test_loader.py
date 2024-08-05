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

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
cf_loader = pytest.importorskip("crossfit.backend.torch.loader")


def test_padding_side_right():
    sample_data_for_padding = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]])
    }

    loader = cf_loader.InMemoryLoader(
        sample_data_for_padding, batch_size=2, max_seq_len=3, padding_side="right"
    )
    batches = list(loader)

    expected_batch_1 = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]])}
    expected_batch_2 = {"input_ids": torch.tensor([[6, 7, 8]])}

    assert len(batches) == 2
    assert torch.equal(batches[0]["input_ids"], expected_batch_1["input_ids"])
    assert torch.equal(batches[1]["input_ids"], expected_batch_2["input_ids"])


def test_padding_side_left():
    sample_data_for_padding = {
        "input_ids": torch.tensor([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6], [0, 6, 7, 8, 9]])
    }

    loader = cf_loader.InMemoryLoader(
        sample_data_for_padding, batch_size=2, max_seq_len=3, padding_side="left"
    )
    batches = list(loader)

    expected_batch_1 = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
    expected_batch_2 = {"input_ids": torch.tensor([[7, 8, 9]])}

    assert len(batches) == 2
    assert torch.equal(batches[0]["input_ids"], expected_batch_1["input_ids"])
    assert torch.equal(batches[1]["input_ids"], expected_batch_2["input_ids"])
