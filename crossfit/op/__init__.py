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

# flake8: noqa

from crossfit.op.base import Op
from crossfit.op.combinators import Sequential

__all__ = [
    "Op",
    "Sequential",
]

try:
    from crossfit.backend.torch.op.embed import Embedder

    __all__.append("Embedder")
except ImportError:
    pass


try:
    from crossfit.backend.torch.op.base import Predictor

    __all__.append("Predictor")
except ImportError:
    pass


try:
    from crossfit.op.tokenize import Tokenizer

    __all__.append("Tokenizer")
except ImportError:
    pass


try:
    from crossfit.op.label import Labeler

    __all__.append("Labeler")
except ImportError:
    pass


try:
    from crossfit.op.vector_search import CuMLANNSearch, CuMLExactSearch, RaftExactSearch

    __all__.extend(["CuMLANNSearch", "CuMLExactSearch", "RaftExactSearch"])
except ImportError:
    pass
