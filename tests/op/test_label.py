import pytest

cudf = pytest.importorskip("cudf")

import crossfit as cf


def test_labeler_basic():
    df = cudf.Series(
        {
            "col1": [
                [0.1, 0.2, 0.5],
                [0.2, 0.1, 0.3],
                [0.3, 0.2, 0.1],
                [0.2, 0.3, 0.1],
            ]
        }
    )
    labeler = cf.op.Labeler(list("abc"))
    results = labeler(df)

    assert results.to_pandas().values.tolist() == ["c", "c", "a", "b"]

def test_labeler_first_axis():
    df = cudf.Series(
        {
            "col1": [
                [0.1, 0.2, 0.5],
                [0.2, 0.1, 0.3],
                [0.3, 0.2, 0.1],
                [0.2, 0.3, 0.1],
            ]
        }
    )
    labeler = cf.op.Labeler(list("abcd"), axis=0)
    results = labeler(df)

    assert results.to_pandas().values.tolist() == ["c", "d", "a"]
