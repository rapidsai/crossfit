# Copyright 2023 NVIDIA CORPORATION
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

import logging

from crossfit.data.array import conversion
from crossfit.data.array.dispatch import ArrayBackend, np_backend_dispatch

try:
    import torch

    # TODO: Use keras.backend.torch instead of torch
    # from keras.backend import torch as keras_torch

    class TorchBackend(ArrayBackend):
        def __init__(self):
            super().__init__(torch)
            # super().__init__(keras_torch.numpy)

    torch_backend = TorchBackend()
    np_backend_dispatch.register(torch.Tensor)(torch_backend)

    @conversion.dispatch_to_dlpack.register(torch.Tensor)
    def torch_to_dlpack(input_array: torch.Tensor):
        logging.debug(f"Converting {input_array} to DLPack")
        return torch.utils.dlpack.to_dlpack(input_array)

    @conversion.dispatch_from_dlpack.register(torch.Tensor)
    def torch_from_dlpack(capsule) -> torch.Tensor:
        logging.debug(f"Converting {capsule} to torch.Tensor")
        return torch.utils.dlpack.from_dlpack(capsule)

    @conversion.dispatch_to_array.register(torch.Tensor)
    def torch_to_array(input_array: torch.Tensor):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return input_array.numpy()

    @conversion.dispatch_from_array.register(torch.Tensor)
    def torch_from_array(array) -> torch.Tensor:
        logging.debug(f"Converting {array} to torch.Tensor")
        return torch.from_numpy(array)

except ImportError:
    torch_backend = None
