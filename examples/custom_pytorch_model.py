import argparse
import os

import dask_cudf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

import crossfit as cf
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel

BATCH_SIZE = 16
NUM_ROWS = 1_000


class CFG:
    model = "sentence-transformers/all-MiniLM-L6-v2"
    fc_dropout = 0.2
    max_len = 512
    out_dim = 3


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.out_dim)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.fc(self.fc_dropout(feature))
        output = torch.softmax(output[:, 0, :], dim=1)
        return output


# The user must provide a load_model function
def load_model(CFG, device, model_path):
    model = CustomModel(CFG, config_path=None, pretrained=True)
    model = model.to(device)

    if os.path.exists(model_path):
        sd = torch.load(os.path.join(model_path), map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
        model.load_state_dict(sd, strict=True)

    model.eval()
    return model


class MyModel(HFModel):
    def load_model(self, device="cuda"):
        return load_model(CFG, device=device, model_path=self.path_or_name)

    def load_cfg(self):
        return AutoConfig.from_pretrained(self.path_or_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="BEIR evaluation using Crossfit")
    parser.add_argument("input_parquet_path", help="Input parquet file")
    parser.add_argument("output_parquet_path", help="Output file")
    parser.add_argument(
        "--input-column", type=str, default="text", help="Column name in input dataframe"
    )
    parser.add_argument("--pool-size", type=str, default="12GB", help="RMM pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    ddf = dask_cudf.read_parquet(args.input_parquet_path)

    model = MyModel(CFG.model)
    labels = ["foo", "bar", "baz"]

    with cf.Distributed(rmm_pool_size=args.pool_size, n_workers=args.num_workers):
        pipe = op.Sequential(
            op.Tokenizer(model, cols=[args.input_column]),
            op.Predictor(model, sorted_data_loader=True, batch_size=args.batch_size),
            op.Labeler(labels, cols=["preds"]),
            repartition=args.partitions,
        )
        outputs = pipe(ddf)
        outputs.to_parquet(args.output_parquet_path)


if __name__ == "__main__":
    main()
