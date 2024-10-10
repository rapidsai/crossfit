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

import gc
from typing import List, Union

import torch
import torch.nn.functional as F


def pad_tensors(
    tensor_list: List[torch.Tensor],
    pad_token_id: Union[int, float] = 0,
    padding_side: str = "right",
) -> List[torch.Tensor]:
    """
    Pad a list of tensors to the same shape.

    This function takes a list of tensors with potentially different shapes and pads them
    to match the largest dimensions across all tensors in the list. The padding is applied
    to the end of each dimension.

    Args:
        tensor_list (List[torch.Tensor]): A list of tensors to be padded.
        pad_token_id (Union[int, float], optional): The value to use for padding. Defaults to 0.

    Returns:
        List[torch.Tensor]: A list of padded tensors, all with the same shape.
    """
    if padding_side not in ["right", "left"]:
        raise ValueError("padding_side must be either 'right' or 'left'")
    # Find the maximum size for each dimension except the batch dimension
    max_sizes = list(tensor_list[0].shape)
    for tensor in tensor_list:
        for dim in range(1, len(tensor.shape)):
            if tensor.shape[dim] > max_sizes[dim]:
                max_sizes[dim] = tensor.shape[dim]

    # Pad each tensor to the maximum size for each dimension
    padded_tensors = []
    for tensor in tensor_list:
        pad_sizes = []
        for i in range(len(tensor.shape) - 1, 0, -1):
            pad_size = max_sizes[i] - tensor.shape[i]
            if padding_side == "right":
                pad_sizes.extend([0, pad_size])
            else:
                pad_sizes.extend([pad_size, 0])

        # Apply padding
        padded_tensor = F.pad(tensor, pad_sizes, mode="constant", value=pad_token_id)
        padded_tensors.append(padded_tensor)

    return padded_tensors


def concat_and_pad_tensors(
    all_outputs_ls: List[torch.Tensor],
    pad_token_id: Union[int, float] = 0,
    padding_side: str = "right",
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
    padded_outputs = pad_tensors(all_outputs_ls, pad_token_id, padding_side)

    # Concatenate the padded tensors
    return torch.cat(padded_outputs, dim=0)


def cleanup_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    return None


def reset_memory_tracking() -> None:
    """
    Resets memory counters.

    This function enables memory usage statistics tracking and resets the counters
    for peak memory usage. It handles both RMM (RAPIDS Memory Manager) and PyTorch's
    native CUDA memory tracking, depending on the current allocator backend.

    If RMM is being used as the allocator, it enables RMM statistics and pushes a new
    statistics context. If the default PyTorch allocator is being used, it resets the
    peak memory stats for CUDA.

    Returns:
        None
    """
    if is_torch_memory_rmm():
        import rmm

        rmm.statistics.enable_statistics()
        rmm.statistics.push_statistics()
    else:
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_used() -> int:
    """
    Get the peak memory usage in bytes.

    This function retrieves the peak memory usage, either from RMM statistics
    if the RMM allocator is being used, or from PyTorch's CUDA memory stats.

    Returns:
        int: Peak memory usage in bytes.
    """
    if is_torch_memory_rmm():
        import rmm

        stats = rmm.statistics.pop_statistics()
        return stats.peak_bytes
    else:
        return torch.cuda.max_memory_allocated()


def is_torch_memory_rmm():
    # TODO: This is hacky, we need to check if the allocator is rmm
    #  and then reset the peak memory stats
    # we get this fixed in Pytorch
    # https://github.com/pytorch/pytorch/issues/133281
    # https://github.com/pytorch/pytorch/issues/133280
    return torch.cuda.memory.get_allocator_backend() == "pluggable"
