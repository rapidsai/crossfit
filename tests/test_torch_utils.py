import pytest

torch = pytest.importorskip("torch")


def test_pad_tensors_2d():
    from crossfit.utils.torch_utils import pad_tensors

    # Test with 2D tensors
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
    tensor_list = [tensor1, tensor2]

    padded_tensors = pad_tensors(tensor_list)

    assert len(padded_tensors) == 2
    assert padded_tensors[0].shape == (3, 3)
    assert padded_tensors[1].shape == (3, 3)
    assert torch.all(padded_tensors[0] == torch.tensor([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
    assert torch.all(padded_tensors[1] == tensor2)


def test_pad_tensors_3d():
    from crossfit.utils.torch_utils import pad_tensors

    # Test with 3D tensors
    tensor1 = torch.rand(2, 3, 4)
    tensor2 = torch.rand(3, 2, 5)
    tensor_list = [tensor1, tensor2]

    padded_tensors = pad_tensors(tensor_list)

    assert len(padded_tensors) == 2
    assert padded_tensors[0].shape == (3, 3, 5)
    assert padded_tensors[1].shape == (3, 3, 5)


def test_pad_tensors_custom_value():
    from crossfit.utils.torch_utils import pad_tensors

    # Test with custom pad value
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6, 7]])
    tensor_list = [tensor1, tensor2]

    padded_tensors = pad_tensors(tensor_list, pad_token_id=-1)

    assert torch.all(padded_tensors[0] == torch.tensor([[1, 2, -1], [3, 4, -1]]))
    assert torch.all(padded_tensors[1] == torch.tensor([[5, 6, 7], [-1, -1, -1]]))


def test_concat_padded_tensors():
    from crossfit.utils.torch_utils import concat_and_pad_tensors

    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6, 7], [8, 9, 10]])
    all_outputs_ls = [tensor1, tensor2]

    result = concat_and_pad_tensors(all_outputs_ls)

    expected_result = torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 7], [8, 9, 10]])

    assert torch.all(result == expected_result)


def test_concat_padded_tensors_custom_value():
    from crossfit.utils.torch_utils import concat_and_pad_tensors

    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6, 7], [8, 9, 10]])
    all_outputs_ls = [tensor1, tensor2]

    result = concat_and_pad_tensors(all_outputs_ls, pad_token_id=-1)

    expected_result = torch.tensor([[1, 2, -1], [3, 4, -1], [5, 6, 7], [8, 9, 10]])

    assert torch.all(result == expected_result)


def test_concat_padded_tensors_different_devices():
    from crossfit.utils.torch_utils import concat_and_pad_tensors

    if torch.cuda.is_available():
        tensor1 = torch.tensor([[1, 2], [3, 4]], device="cuda")
        tensor2 = torch.tensor([[5, 6, 7], [8, 9, 10]], device="cpu")
        all_outputs_ls = [tensor1, tensor2]

        result = concat_and_pad_tensors(all_outputs_ls)

        assert result.device == tensor1.device
        assert result.shape == (4, 3)
    else:
        pytest.skip("CUDA not available, skipping test_concat_padded_tensors_different_devices")
