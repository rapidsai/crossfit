import logging

from crossfit.array import conversion
from crossfit.array.dispatch import np_backend_dispatch, NPBackend


try:
    import torch

    class TorchBackend(NPBackend):
        def __init__(self):
            super().__init__(torch)

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
