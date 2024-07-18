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


@np_backend_dispatch.register_lazy("cudf")
def register_cudf_backend():
    import cudf

    class CudfBackend(ArrayBackend):
        def __init__(self):
            super().__init__(cudf)

        def concatenate(self, series_list, *, axis=None):
            return cudf.concat(series_list, axis=axis or 0)

    np_backend_dispatch.register((cudf.Series, cudf.Index))(CudfBackend())


@conversion.dispatch_to_dlpack.register_lazy("cudf")
def register_cudf_to_dlpack():
    import cudf

    @conversion.dispatch_to_dlpack.register(cudf.Series)
    def cudf_to_dlpack(input_array: cudf.Series):
        logging.debug(f"Converting {input_array} to DLPack")

        if input_array.dtype.name == "list":
            if not input_array.list.len().min() == input_array.list.len().max():
                raise NotImplementedError("Cannot convert list column with variable length")

            dim = input_array.list.len().iloc[0]
            return input_array.list.leaves.values.reshape(-1, dim).toDlpack()

        return input_array.to_dlpack()


@conversion.dispatch_from_dlpack.register_lazy("cudf")
def register_cudf_from_dlpack():
    import cudf

    @conversion.dispatch_from_dlpack.register(cudf.Series)
    def cudf_from_dlpack(capsule) -> cudf.Series:
        logging.debug(f"Converting {capsule} to cudf.Series")
        return cudf.io.from_dlpack(capsule)


@conversion.dispatch_to_array.register_lazy("cudf")
def register_cudf_to_array():
    import cudf

    @conversion.dispatch_to_array.register(cudf.Series)
    def cudf_to_array(input_array: cudf.Series):
        logging.debug(f"Converting {input_array} to np.ndarray")
        return input_array.to_numpy()


@conversion.dispatch_from_array.register_lazy("cudf")
@conversion.dispatch_from_cuda_array.register_lazy("cudf")
def register_cudf_from_array():
    import cudf

    @conversion.dispatch_from_array.register(cudf.Series)
    @conversion.dispatch_from_cuda_array.register(cudf.Series)
    def cudf_from_array(array) -> cudf.Series:
        logging.debug(f"Converting {array} to cudf.Series")
        return cudf.Series(array)
