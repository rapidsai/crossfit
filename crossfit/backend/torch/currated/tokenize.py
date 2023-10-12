from typing import Optional

import cudf
import cupy as cp
import torch
from cudf.core.subword_tokenizer import _cast_to_appropriate_type
from curated_transformers.tokenizers import Tokenizer

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
        max_length: Optional[int] = None,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_strings(self, sentences):
        pieces = self.tokenizer(sentences.to_arrow().to_pylist())

        ids = pieces.padded_tensor(pad_left=True, device="cuda")
        attention_mask = pieces.attention_mask(
            pad_left=True, device="cuda"
        ).bool_mask.to(torch.int8)

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
