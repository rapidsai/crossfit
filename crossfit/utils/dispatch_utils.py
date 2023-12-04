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


def supports(dispatch: Dispatch) -> set:
    if dispatch._lazy:
        _to_remove = set()
        for module in dispatch._lazy:
            try:
                register = dispatch._lazy[module]
            except KeyError:
                pass
            else:
                try:
                    register()
                    _to_remove.add(module)
                except ModuleNotFoundError:
                    pass
        for module in _to_remove:
            dispatch._lazy.pop(module, None)

    return set(dispatch._lookup)
