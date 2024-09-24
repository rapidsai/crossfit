import pytest

cudf = pytest.importorskip("cudf")
torch = pytest.importorskip("torch")

from crossfit.backend.cudf.series import (  # noqa: E402
    create_list_series_from_1d_or_2d_ar,
    create_nested_list_series_from_3d_ar,
)


@pytest.mark.singlegpu
def test_create_list_series_from_1d_or_2d_ar_1d():
    tensor = torch.tensor([101, 102, 103])
    index = [1, 2, 3]
    series = create_list_series_from_1d_or_2d_ar(tensor, index)
    assert isinstance(series, cudf.Series)
    expected = cudf.Series([[101], [102], [103]], index=index)
    # convert to pandas because cudf.Series.equals doesn't work for list series
    assert series.to_pandas().equals(expected.to_pandas())


@pytest.mark.singlegpu
def test_create_list_series_from_1d_or_2d_ar_2d():
    tensor = torch.tensor([[101, 102], [103, 104], [105, 106]])
    index = [1, 2, 3]
    series = create_list_series_from_1d_or_2d_ar(tensor, index)
    assert isinstance(series, cudf.Series)
    expected = cudf.Series([[101, 102], [103, 104], [105, 106]], index=index)
    # convert to pandas because cudf.Series.equals doesn't work for list series
    assert series.to_pandas().equals(expected.to_pandas())


@pytest.mark.singlegpu
def test_create_nested_list_series_from_3d_ar():
    nested_list = [[[101, 102], [103, 104], [105, 106]], [[201, 202], [203, 204], [205, 206]]]
    tensor = torch.tensor(nested_list)
    index = [1, 2]
    series = create_nested_list_series_from_3d_ar(tensor, index)
    assert isinstance(series, cudf.Series)
    expected = cudf.Series(nested_list, index=index)
    # convert to pandas because cudf.Series.equals doesn't work for list series
    assert series.to_pandas().equals(expected.to_pandas())
