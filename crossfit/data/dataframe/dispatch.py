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

from dask.utils import Dispatch


class _CrossFrameDispatch(Dispatch):
    def __call__(self, data, *args, **kwargs):
        from crossfit.data.dataframe.core import FrameBackend

        if isinstance(data, FrameBackend):
            return data

        # TODO: Fix this
        from crossfit.backend.dask.dataframe import DaskDataFrame
        from crossfit.backend.pandas.dataframe import PandasDataFrame

        backends = [PandasDataFrame, DaskDataFrame]

        try:
            from crossfit.backend.cudf.dataframe import CudfDataFrame

            CudfDataFrame._lib()
            backends.append(CudfDataFrame)
        except ImportError:
            pass

        for backend in backends:
            if isinstance(data, getattr(backend._lib(), "DataFrame")):
                return backend(data, *args, **kwargs)

        return super().__call__(data, *args, **kwargs)


CrossFrame = _CrossFrameDispatch(name="frame_dispatch")
