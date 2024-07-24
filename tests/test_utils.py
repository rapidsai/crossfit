# Copyright 2024 NVIDIA CORPORATION
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

import pytest

from crossfit.utils.model_adapter import adapt_model_input

torch = pytest.importorskip("torch")
sentence_transformers = pytest.importorskip("sentence_transformers")
transformers = pytest.importorskip("transformers")


def test_adapt_model_input_hf():
    from transformers import AutoTokenizer, DistilBertModel

    with torch.no_grad():
        model_hf = DistilBertModel.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        # Hugging Face model output
        outputs_hf = model_hf(**inputs)
        adapted_inputs_hf = adapt_model_input(model_hf, inputs)
        assert torch.equal(adapted_inputs_hf.last_hidden_state, outputs_hf.last_hidden_state)


def test_adapt_model_input_sentence_transformers():
    from transformers import AutoTokenizer

    with torch.no_grad():
        model_st = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2").to("cpu")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        inputs = tokenizer(
            ["Hello", "my dog is cute"], return_tensors="pt", padding=True, truncation=True
        )
        # Sentence Transformers model output
        expected_output = model_st(inputs)
        adapted_output_st = adapt_model_input(model_st, inputs)

        assert torch.equal(adapted_output_st.sentence_embedding, expected_output.sentence_embedding)
