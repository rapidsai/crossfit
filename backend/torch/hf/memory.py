from crossfit.backend.torch.loader import MemoryEstimator
from dask.utils import Dispatch

from transformers import BertConfig


class HFMemoryEstimator(MemoryEstimator):
    registry = Dispatch(name="hf_memory_estimator")

    def __init__(self, max_mem_gb, cfg):
        super().__init__(max_mem_gb)
        self.cfg = cfg

    def max_seq_length(self) -> int:
        return self.cfg.max_position_embeddings

    def estimate(self, max_num_tokens: int, batch_size: int) -> int:
        return self.registry(self.cfg, max_num_tokens, batch_size)


@HFMemoryEstimator.registry.register(BertConfig)
def estimate(cfg, max_num_tokens: int, batch_size: int) -> int:
    """
    Estimates the memory consumption for a given SentenceTransformer config and batch.
    Args:
        cfg (transformers.AutoConfig): Configuration object for SentenceTransformer model
        max_num_tokens (int): Maximum number of tokens in a batch
        batch_size (int): Batch size

    Returns:
        total_memory_gb (float): Estimated total memory in gigabytes
    """

    # Model parameters
    total_params = _count_parameters_from_bert_config(cfg)
    model_size = total_params * 4  # in bytes (float32 for each parameter)

    # Hidden size and number of layers
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers

    # Activation memory: assuming a forward and backward pass (even if it's not training, for worst case)
    # The '3' includes memory for input, output and intermediate states for each layer.
    activation_memory = num_layers * batch_size * max_num_tokens * hidden_size * 4 * 3  # in bytes

    # Sum up
    total_memory = model_size + activation_memory  # in bytes
    total_memory_gb = total_memory / (1024**3)  # Convert to GBs

    return total_memory_gb


def _count_parameters_from_bert_config(config):
    # Embedding Layer
    embedding_params = (
        config.vocab_size * config.hidden_size
        + config.max_position_embeddings * config.hidden_size
        + config.type_vocab_size * config.hidden_size
    )
    embedding_params += config.hidden_size * 2  # LayerNorm

    # Parameters in one transformer layer
    attention_params = (config.hidden_size * 3) * config.num_attention_heads + (
        config.num_attention_heads * config.hidden_size
    )
    attention_params += config.hidden_size * 2  # LayerNorm
    ffnn_params = (config.hidden_size * config.intermediate_size + config.intermediate_size) + (
        config.intermediate_size * config.hidden_size + config.hidden_size
    )
    ffnn_params += config.hidden_size * 2  # LayerNorm

    # Parameters in all transformer layers
    transformer_params = (attention_params + ffnn_params) * config.num_hidden_layers

    # Total parameters
    total_params = embedding_params + transformer_params

    return total_params
