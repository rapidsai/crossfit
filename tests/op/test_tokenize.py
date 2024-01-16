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
