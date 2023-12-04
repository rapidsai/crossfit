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

from crossfit.dataset.beir import load as beir


def load_dataset(name, out_dir=None, blocksize=2**30, overwrite=False, tiny_sample=False):
    load_fn_name = "load_dataset" if not tiny_sample else "load_test_dataset"

    if name.startswith("beir/"):
        return getattr(beir, load_fn_name)(
            name[len("beir/") :],
            out_dir=out_dir,
            blocksize=blocksize,
            overwrite=overwrite,
        )

    raise NotImplementedError(f"Unknown dataset: {name}")
