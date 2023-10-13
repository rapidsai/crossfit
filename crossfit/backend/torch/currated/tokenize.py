from typing import Optional

import cudf
import cupy as cp
import torch
from curated_transformers.tokenizers import Tokenizer, AutoTokenizer

from crossfit.backend.cudf.series import create_list_series_from_2d_ar
from crossfit.data.array.conversion import convert_array
from crossfit.op.tokenize import _TokenizerOp, clip_tokens


class CurratedTokenizer(_TokenizerOp):
    def __init__(
        self,
        tokenizer: Tokenizer,
        cols=None,
        keep_cols=None,
        pre=None,
        pad_left=False,
        max_length: Optional[int] = None,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_left = pad_left

    @classmethod
    def from_hf_hub(
        cls,
        *,
        name: str,
        cols=None,
        revision: str = "main",
    ) -> "CurratedTokenizer":
        tokenizer = AutoTokenizer.from_hf_hub(name=name, revision=revision)

        return cls(tokenizer, cols=cols)

    def tokenize_strings(self, sentences):
        pieces = self.tokenizer(sentences.to_arrow().to_pylist())
        ids = pieces.padded_tensor(device="cuda", pad_left=self.pad_left)
        mask = pieces.attention_mask(device="cuda", pad_left=self.pad_left)
        attention_mask = mask.bool_mask.to(torch.int8)

        return {
            "input_ids": convert_array(ids, cp.ndarray),
            "attention_mask": convert_array(attention_mask, cp.ndarray),
        }

    def call_column(self, data):
        if isinstance(data, cudf.DataFrame):
            raise ValueError(
                "data must be a Series, got DataFrame. Add a pre step to convert to Series"
            )

        text = data
        tokenized_data = self.tokenize_strings(text)
        if self.max_length:
            tokenized_data = clip_tokens(tokenized_data, max_length=self.max_length)

        input_ids = create_list_series_from_2d_ar(
            tokenized_data["input_ids"].astype("int32"), data.index
        )
        attention_mask = create_list_series_from_2d_ar(
            tokenized_data["attention_mask"].astype("int32").reshape(len(data), -1),
            data.index,
        )

        return input_ids, attention_mask
