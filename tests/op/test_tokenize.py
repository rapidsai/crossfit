import pytest

cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")

import crossfit as cf  # noqa: E402
from crossfit import op  # noqa: E402


@pytest.mark.singlegpu
def test_token_counter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    df = cudf.DataFrame(
        {
            "text": [
                "!",
                "query: how much protein should a female eat",
                "query: summit define",
                "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",  # noqa: E501
                "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",  # noqa: E501
            ]
        }
    )

    ddf = dask_cudf.from_cudf(df, npartitions=2)

    model = cf.SentenceTransformerModel(model_name)

    pipe = op.Sequential(
        op.Tokenizer(model, cols=["text"]),
        op.TokenCounter(cols=["input_ids"]),
    )

    num_tokens = pipe(ddf).compute()
    expected = cudf.DataFrame(
        {
            "token_count": cudf.Series([3, 11, 6, 75, 50], dtype="int32")
        }
    )

    cudf.testing.testing.assert_frame_equal(num_tokens, expected)
