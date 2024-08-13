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

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")
dd = pytest.importorskip("dask.dataframe")
pd = pytest.importorskip("pandas")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

import crossfit as cf  # noqa: E402
from crossfit import op  # noqa: E402

cf_loader = pytest.importorskip("crossfit.backend.torch.loader")


def test_tokenizer_sentence_piece(model_name="microsoft/deberta-v3-base"):
    model = cf.HFModel(model_name)
    tokenizer = op.Tokenizer(model, cols=["text"], tokenizer_type="spm")
    input_strings = ["hello world", "this is a sentence"]
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"text": input_strings}),
        npartitions=1,
    )
    results = tokenizer(ddf)
    results = results.compute()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenized_strings = hf_tokenizer.batch_encode_plus(
        input_strings, return_tensors="pt", padding="longest"
    )
    assert isinstance(results, cudf.DataFrame)
    np.testing.assert_equal(
        np.asarray(results["input_ids"][0]), tokenized_strings["input_ids"][0].numpy()
    )
    np.testing.assert_equal(
        np.asarray(results["input_ids"][1]), tokenized_strings["input_ids"][1].numpy()
    )


def test_tokenizer_max_chars(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = cf.SentenceTransformerModel(model_name)
    tokenizer1 = op.Tokenizer(model, cols=["text"], max_chars=7)
    ddf1 = dask_cudf.from_cudf(
        cudf.DataFrame({"text": ["hello world", "this is a sentence"]}),
        npartitions=2,
    )
    results1 = tokenizer1(ddf1)
    results1 = results1.compute()

    tokenizer2 = op.Tokenizer(model, cols=["text"], max_chars=None)
    ddf2 = dask_cudf.from_cudf(
        cudf.DataFrame({"text": ["hello world"[:7], "this is a sentence"[:7]]}),
        npartitions=2,
    )
    results2 = tokenizer2(ddf2)
    results2 = results2.compute()

    assert results1["input_ids"][0] == results2["input_ids"][0]
    assert results1["input_ids"][1] == results2["input_ids"][1]


def test_tokenizer_padded(model_name="microsoft/deberta-v3-base"):
    model = cf.HFModel(model_name)
    tokenizer = op.Tokenizer(model, cols=["text"], tokenizer_type="spm")
    input_strings = ["hello world", "this is a sentence"]
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"text": input_strings}),
        npartitions=1,
    )
    results = tokenizer(ddf)
    results = results.compute()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenized_strings = hf_tokenizer.batch_encode_plus(
        input_strings, return_tensors="pt", padding="longest"
    )
    assert isinstance(results, cudf.DataFrame)
    np.testing.assert_equal(
        np.asarray(results["input_ids"][0]), tokenized_strings["input_ids"][0].numpy()
    )
    np.testing.assert_equal(
        np.asarray(results["input_ids"][1]), tokenized_strings["input_ids"][1].numpy()
    )


def test_clip_tokens_right_padding():
    input_ids = cp.array([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
    attention_mask = cp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = cf_loader.clip_tokens(token_o, max_length=4, padding_side="right", pad_token_id=0)

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 4)
    assert result["attention_mask"].shape == (2, 4)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]]))
    assert torch.equal(
        result["attention_mask"].to("cpu"), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    )


def test_clip_tokens_left_padding():
    input_ids = cp.array([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    attention_mask = cp.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = cf_loader.clip_tokens(token_o, max_length=4, padding_side="left", pad_token_id=0)

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 4)
    assert result["attention_mask"].shape == (2, 4)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert torch.equal(
        result["attention_mask"].to("cpu"), torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    )


def test_clip_tokens_no_clipping_needed():
    input_ids = cp.array([[1, 2, 3], [4, 5, 6]])
    attention_mask = cp.array([[1, 1, 1], [1, 1, 1]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = cf_loader.clip_tokens(token_o, max_length=4, padding_side="right", pad_token_id=0)

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 3)
    assert result["attention_mask"].shape == (2, 3)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(result["attention_mask"].to("cpu"), torch.tensor([[1, 1, 1], [1, 1, 1]]))


def test_tokenize_strings_cpu(model_name="microsoft/deberta-v3-base"):
    model = cf.HFModel(model_name)
    tokenizer = op.Tokenizer(model, cols=["text"], tokenizer_type="spm")
    input_strings = ["hello world", "this is a sentence"]
    ddf = dd.from_pandas(pd.DataFrame({"text": input_strings}), npartitions=1)
    results = tokenizer(ddf)
    results = results.compute()
