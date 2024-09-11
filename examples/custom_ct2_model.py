import os
import argparse
from dataclasses import dataclass
from functools import lru_cache
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
import dask_cudf
import crossfit as cf

import ctranslate2
from transformers import AutoConfig, AutoTokenizer

@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    ct2_model_path: str
    string_tok_inf: bool = True

class CT2CustomModel():
    def __init__(self, config: TranslationConfig, device="cuda"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code = True,
        )
        self.model = ctranslate2.Translator(model_path=config.ct2_model_path, device=device)
        self.string_tok_inf = config.string_tok_inf

    def clean_extra_tokens(self, token_2d):
        results=[]
        for token_1d in token_2d:
            result = []
            for t in token_1d:
                if t==self.tokenizer.pad_token or t==self.tokenizer.bos_token or t==self.tokenizer.eos_token or t==self.tokenizer.unk_token:
                    pass
                else:
                    result.append(t)
            results.append(result)
        return results

    def __call__(self, batch):
        token_ids_2d=batch['input_ids']
        token_ids_1d = token_ids_2d.view(-1).tolist()
        tokens_1d = self.tokenizer.convert_ids_to_tokens(token_ids_1d)
        tokens_2d = [tokens_1d[i:i + token_ids_2d.size(1)] for i in range(0, len(tokens_1d), token_ids_2d.size(1))]
        tokenss = self.clean_extra_tokens(tokens_2d)

        tr_res = self.model.translate_batch(
            tokenss,
            min_decoding_length=0,
            max_decoding_length=256,
            beam_size=5,
            num_hypotheses=1,
        )
        translations = ["".join(x.hypotheses[0]) for x in tr_res]
        return translations

class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.trans_config = config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.string_tok_inf = config.string_tok_inf
        super().__init__(self.trans_config.pretrained_model_name_or_path)

    def load_model(self, device="cuda"):
        model = CT2CustomModel(
            self.trans_config
        )
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_source_positions

    @lru_cache(maxsize=1)
    def load_cfg(self):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Model Predictions using Crossfit")
    parser.add_argument("input_parquet_path", help="Input parquet file")
    parser.add_argument("output_parquet_path", help="Output file")
    parser.add_argument(
        "--ct2-model-dir",
        help="Directory where ctranslate2 optimized model is present",
    )
    parser.add_argument(
        "--input-column", type=str, default="text", help="Column name in input dataframe"
    )
    parser.add_argument("--pool-size", type=str, default="12GB", help="RMM pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ai4bharat/indictrans2-en-indic-1B",
        help="Model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    Config = TranslationConfig(
        pretrained_model_name_or_path=args.model_name,
        ct2_model_path=args.ct2_model_dir,
    )
    ddf = dask_cudf.read_parquet(args.input_parquet_path)

    with cf.Distributed(rmm_pool_size=args.pool_size, n_workers=args.num_workers):
        model = CT2CustomModel(Config)
        pipe = op.Sequential(
            op.Tokenizer(model, cols=[args.input_column], tokenizer_type="sentencepiece", max_length=255),
            op.Predictor(model, sorted_data_loader=True, batch_size=args.batch_size),
            repartition=args.partitions,
            keep_cols=[args.input_column],
        )
        outputs = pipe(ddf)
        outputs.to_parquet(args.output_parquet_path)


if __name__ == "__main__":
    main()