from typing import List, Union

import torch
import torch.nn.functional as F


def pad_tensors(
    tensor_list: List[torch.Tensor], pad_token_id: Union[int, float] = 0
) -> List[torch.Tensor]:
    """
    Pad a list of tensors to the same shape.

    This function takes a list of tensors with potentially different shapes and pads them
    to match the largest dimensions across all tensors in the list. The padding is applied
    to the end of the last dimension.

    Args:
        tensor_list (List[torch.Tensor]): A list of tensors to be padded.
        pad_token_id (Union[int, float], optional): The value to use for padding. Defaults to 0.

    Returns:
        List[torch.Tensor]: A list of padded tensors, all with the same shape.
    """
    # Find the maximum size of the last dimension
    max_last_dim = max(tensor.shape[-1] for tensor in tensor_list)

    # Pad each tensor to the maximum size of the last dimension
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = max_last_dim - tensor.shape[-1]
        pad_sizes = [0, pad_size]  # Only pad the last dimension

        # For multi-dimensional tensors, prepend zeros for the other dimensions
        if len(tensor.shape) > 2:
            pad_sizes = [0, 0] * (len(tensor.shape) - 1) + pad_sizes

        padded_tensor = F.pad(tensor, pad_sizes, mode="constant", value=pad_token_id)
        padded_tensors.append(padded_tensor)

    return padded_tensors


def concat_and_pad_tensors(
    all_outputs_ls: List[torch.Tensor], pad_token_id: Union[int, float] = 0
) -> torch.Tensor:
    """
    Concatenate a list of tensors after padding them to the same shape.

    This function first pads all input tensors to the same shape using the `pad_tensors`
    function, then concatenates them along the first dimension.

    Args:
        all_outputs_ls (List[torch.Tensor]): A list of tensors to be padded and concatenated.
        pad_token_id (Union[int, float], optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: A single tensor resulting from the concatenation of all padded input tensors.

    """
    # Ensure all tensors are on the same device
    device = all_outputs_ls[0].device
    all_outputs_ls = [tensor.to(device) for tensor in all_outputs_ls]

    # Pad the tensors
    padded_outputs = pad_tensors(all_outputs_ls, pad_token_id)

    # Concatenate the padded tensors
    return torch.cat(padded_outputs, dim=0)
