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
class Model:
    def __init__(self, path_or_name: str, max_mem_gb: int = 16, model_output_type: str = "numeric"):
        self.path_or_name = path_or_name
        self.max_mem_gb = max_mem_gb
        if model_output_type == "numeric" or model_output_type == "string":
            self.model_output_type = model_output_type
        else:
            raise ValueError("Invalid model output type provided. Allowed values are : 'string' or 'numeric'.")

    def load_model(self, device="cuda"):
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def load_on_worker(self, worker):
        raise NotImplementedError()

    def unload_from_worker(self, worker):
        raise NotImplementedError()

    def call_on_worker(self, worker, *args, **kwargs):
        return worker.torch_model(*args, **kwargs)

    def get_model(self, worker):
        if not hasattr(worker, "torch_model"):
            self.load_on_worker(worker)
        return worker.torch_model

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        raise NotImplementedError()

    def max_seq_length(self) -> int:
        raise NotImplementedError()
