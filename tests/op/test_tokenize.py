import pytest

cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")
transformers = pytest.importorskip("transformers")

import crossfit as cf  # noqa: E402
from crossfit import op  # noqa: E402


def test_tokenizer_sentence_piece(model_name="microsoft/deberta-v3-base"):
    model = cf.HFModel(model_name)
    tokenizer = op.Tokenizer(model, cols=["text"], tokenizer_type="spm")
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"text": ["hello world", "this is a sentence"]}),
        npartitions=2,
    )
    results = tokenizer(ddf)
    results = results.compute()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    assert isinstance(results, cudf.DataFrame)
    assert results["input_ids"][0] == hf_tokenizer(["hello world"])["input_ids"][0]
    assert results["input_ids"][1] == hf_tokenizer(["this is a sentence"])["input_ids"][0]


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
    ddf = dask_cudf.from_cudf(
        cudf.DataFrame({"text": ["hello world", "this is a sentence"]}),
        npartitions=2,
    )
    results = tokenizer(ddf)
    results = results.compute()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    assert isinstance(results, cudf.DataFrame)
    assert results["input_ids"][0] == hf_tokenizer(["hello world"])["input_ids"][0]
    assert results["input_ids"][1] == hf_tokenizer(["this is a sentence"])["input_ids"][0]
