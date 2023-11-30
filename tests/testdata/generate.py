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

import shutil
from pathlib import Path

from crossfit.dataset.beir.raw import download_all_sampled

current_file_path = Path(__file__).parent.resolve()


if __name__ == "__main__":
    download_all_sampled(out_dir=str(current_file_path))

    # Path to the 'sampled' directory
    sampled_dir = current_file_path / "sampled"

    # Path to the 'beir' directory
    beir_dir = current_file_path / "beir"

    # Remove the 'beir' directory if it already exists
    if beir_dir.exists():
        shutil.rmtree(beir_dir)

    # Rename the 'sampled' directory to 'beir'
    sampled_dir.rename(beir_dir)
