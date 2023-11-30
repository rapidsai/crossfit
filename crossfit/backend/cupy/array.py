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


@np_backend_dispatch.register_lazy("cupy")
def register_cupy_backend():
    import cupy as cp

    class CupyBackend(ArrayBackend):
        def __init__(self):
            super().__init__(cp)

    np_backend_dispatch.register(cp.ndarray)(CupyBackend())


@conversion.dispatch_to_dlpack.register_lazy("cupy")
def register_cupy_to_dlpack():
    import cupy as cp

    @conversion.dispatch_to_dlpack.register(cp.ndarray)
    def cupy_to_dlpack(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to DLPack")
        try:
            return input_array.to_dlpack()
        except AttributeError:
            return input_array.toDlpack()


@conversion.dispatch_from_dlpack.register_lazy("cupy")
def register_cupy_from_dlpack():
    import cupy as cp

    @conversion.dispatch_from_dlpack.register(cp.ndarray)
    def cupy_from_dlpack(capsule) -> cp.ndarray:
        logging.debug(f"Converting {capsule} to cp.ndarray")
        try:
            return cp.from_dlpack(capsule)
        except AttributeError:
            return cp.fromDlpack(capsule)


@conversion.dispatch_to_array.register_lazy("cupy")
def register_cupy_to_array():
    import cupy as cp

    @conversion.dispatch_to_array.register(cp.ndarray)
    def cupy_to_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return cp.asnumpy(input_array)


@conversion.dispatch_from_array.register_lazy("cupy")
@conversion.dispatch_from_cuda_array.register_lazy("cupy")
def register_cupy_from_array():
    import cupy as cp

    @conversion.dispatch_from_array.register(cp.ndarray)
    @conversion.dispatch_from_cuda_array.register(cp.ndarray)
    def cupy_from_array(array) -> cp.ndarray:
        logging.debug(f"Converting {array} to cp.ndarray")
        return cp.asarray(array)


@conversion.dispatch_to_cuda_array.register_lazy("cupy")
def register_cudf_to_cuda_array():
    import cupy as cp

    @conversion.dispatch_to_cuda_array.register(cp.ndarray)
    def cudf_to_cuda_array(input_array: cp.ndarray):
        logging.debug(f"Converting {input_array} to cp.ndarray")
        return input_array
