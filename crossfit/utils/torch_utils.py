import torch
import torch.nn.functional as F


def pad_tensors(tensor_list, pad_token_id=0):
    # Find the maximum dimensions
    max_dims = [
        max(tensor.shape[i] for tensor in tensor_list) for i in range(len(tensor_list[0].shape))
    ]

    # Pad each tensor to the maximum dimensions
    padded_tensors = []
    for tensor in tensor_list:
        pad_sizes = []
        for i in range(len(max_dims) - 1, -1, -1):  # Reverse order for F.pad
            pad_size = max_dims[i] - tensor.shape[i]
            pad_sizes.extend([0, pad_size])

        padded_tensor = F.pad(tensor, pad_sizes, mode="constant", value=pad_token_id)
        padded_tensors.append(padded_tensor)

    return padded_tensors


def concat_and_pad_tensors(all_outputs_ls, pad_token_id=0):
    # Ensure all tensors are on the same device
    device = all_outputs_ls[0].device
    all_outputs_ls = [tensor.to(device) for tensor in all_outputs_ls]

    # Pad the tensors
    padded_outputs = pad_tensors(all_outputs_ls, pad_token_id)

    # Concatenate the padded tensors
    return torch.cat(padded_outputs, dim=0)
