import pytest

cudf = pytest.importorskip("cudf")

import crossfit as cf


def test_labeler():
    df = cudf.Series(
        {"col1": [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.1],
        ]}
    )
    labeler = cf.op.Labeler(list("abc"))
    results = labeler(df)

    assert results.to_pandas().values.tolist() == ["c", "b", "a"]
