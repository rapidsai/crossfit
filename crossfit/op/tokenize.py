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

import os

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer, _cast_to_appropriate_type
from cudf.utils.hash_vocab_utils import hash_vocab
from transformers import AutoConfig, AutoTokenizer

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.backend.torch.model import Model
from crossfit.dataset.home import CF_HOME
from crossfit.op.base import Op


class Tokenizer(Op):
    def __init__(
        self,
        model: Model,
        cols=None,
        keep_cols=None,
        pre=None,
        max_length: int = 1024,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model = model
        self.max_length = max_length

        # Make sure we download the tokenizer just once
        GPUTokenizer.from_pretrained(self.model)

    def setup(self):
        self.tokenizer = GPUTokenizer.from_pretrained(self.model)

    def tokenize_strings(self, sentences, max_length=None):
        return self.tokenizer(
            sentences,
            max_length=max_length or self.max_length,
            max_num_rows=len(sentences),
            padding="max_length",
            return_tensors="cp",
            truncation=True,
            add_special_tokens=True,
        )

    def call_column(self, data):
        if isinstance(data, cudf.DataFrame):
            raise ValueError(
                "data must be a Series, got DataFrame. Add a pre step to convert to Series"
            )

        text = data.replace("", "unknown")
        tokenized_data = self.tokenize_strings(text).copy()
        tokenized_data = clip_tokens(tokenized_data, max_length=self.max_length, return_type="cp")

        input_ids = create_list_series_from_2d_ar(
            tokenized_data["input_ids"].astype("int32"), data.index
        )
        attention_mask = create_list_series_from_2d_ar(
            tokenized_data["attention_mask"].astype("int32"), data.index
        )

        return input_ids, attention_mask

    def call(self, data):
        output = cudf.DataFrame()

        if self.cols is None:
            if not isinstance(data, cudf.Series):
                raise ValueError("data must be a cudf Series")

            input_ids, attention_mask = self.call_column(data)
            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask

            return output

        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")

            input_ids, attention_mask = self.call_column(data[col])
            output[self._construct_name(col, "input_ids")] = input_ids
            output[self._construct_name(col, "attention_mask")] = attention_mask

        return output

    def meta(self):
        tokenized = {
            "input_ids": "int32",
            "attention_mask": "int32",
        }

        if len(self.cols) > 1:
            tokenized = {
                self._construct_name(col, suffix): dtype
                for col in self.cols
                for suffix, dtype in tokenized.items()
            }

        return tokenized

    def _construct_name(self, col_name, suffix):
        if len(self.cols) == 1:
            return suffix

        return f"{col_name}_{suffix}"


class GPUTokenizer(SubwordTokenizer):
    def __init__(self, hash_file: str, do_lower_case: bool = True, config=None):
        super().__init__(str(hash_file), do_lower_case=do_lower_case)
        self.config = config or {"_name_or_path": hash_file}

    @classmethod
    def get_tokenizer_config(cls, name):
        if isinstance(name, Model):
            config = name.load_cfg()
        else:
            config = AutoConfig.from_pretrained(name)
        return config.to_dict()

    @classmethod
    def from_pretrained(cls, name, cache_dir=None):
        # Set default cache_dir if not provided
        if cache_dir is None:
            cache_dir = os.path.join(CF_HOME, "vocab")

        # Create cache_dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Get tokenizer class name from config
        config = cls.get_tokenizer_config(name)
        tokenizer_class = config.get("tokenizer_class")

        # Construct hashed vocab file path
        if not tokenizer_class:
            tokenizer_class = cls.__name__

        hashed_vocab_path = os.path.join(cache_dir, f"{tokenizer_class}-hash.txt")

        # Check if hashed vocab file exists
        if not os.path.exists(hashed_vocab_path):
            # Download and cache the tokenizer from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

            # Save vocabulary to disk
            # `save_vocabulary()` automatically appends `-vocab.txt` suffix.
            vocab_path = tokenizer.save_vocabulary(cache_dir, f"{tokenizer_class}")[0]

            # Hash the vocabulary and save it
            hash_vocab(vocab_path, hashed_vocab_path)

        return cls(hashed_vocab_path, config=config)


def clip_tokens(token_o, max_length, return_type="pt"):
    clip_len = max_length - int((token_o["input_ids"][:, ::-1] != 0).argmax(1).min())
    token_o["input_ids"] = _cast_to_appropriate_type(
        token_o["input_ids"][:, :clip_len], return_type
    )
    token_o["attention_mask"] = _cast_to_appropriate_type(
        token_o["attention_mask"][:, :clip_len], return_type
    )

    if "metadata" in token_o:
        del token_o["metadata"]

    return token_o
