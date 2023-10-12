import dataclasses
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple

import torch
import torch.nn.functional as F
from curated_transformers.generation.auto_generator import AutoGenerator
from curated_transformers.generation.config import (
    GreedyGeneratorConfig,
    SampleGeneratorConfig,
)
from curated_transformers.generation.generator import Generator
from curated_transformers.generation.stop_conditions import StopCondition
from curated_transformers.generation.string_generator import StringGenerator
from curated_transformers.layers.attention import AttentionMask
from curated_transformers.layers.cache import KeyValueCache
from curated_transformers.models.output import CacheT

# from curated_transformers.layers.attention import enable_torch_sdp
from curated_transformers.quantization.bnb import BitsAndBytesConfig
from tqdm import tqdm

from crossfit.backend.torch.loader import InMemoryLoader
from crossfit.op.base import Op


class CuratedGenerator(Op):
    def __init__(
        self,
        model_name: str,
        config=GreedyGeneratorConfig(),
        pre=None,
        cols=False,
        keep_cols=None,
    ):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.model_name = model_name
        self.config = config

    def setup(self):
        quantization_config = BitsAndBytesConfig.for_4bit()

        self.generator = AutoGenerator.from_hf_hub(
            name=self.model_name,
            device=torch.device("cuda"),
            quantization_config=quantization_config,
        )
        self.config = get_config(self.generator, self.config)

    def call(self, data, partition_info=None):
        model = self.generator.generator.inner.model
        tokenizer = self.generator.generator.tokenizer

        outputs = PartitionGenerator(tokenizer, BatchGenerator(model))(
            data, config=self.config
        )

        # TODO: Implement the rest


class BatchGenerator(Generator):
    def generate_df(
        self,
        *,
        attention_mask,
        ids,
        config,
        index=None,
        generated_ids=None,
        cache=None,
        max_steps=10,
    ):
        self.model.eval()

        logits_transform = config.logits_transform()
        stop_condition = config.stop_condition()
        if isinstance(config, GreedyGeneratorConfig):
            generation_step = self._decode_greedy
        elif isinstance(config, SampleGeneratorConfig):
            generation_step = self._decode_sample
        else:
            raise ValueError(
                f"Unknown generator configuration: {type(config).__name__}"
            )

        state = CuDFGeneratorState(
            attention_mask=attention_mask, cache=cache, prompt_ids=ids, index=index
        )
        if generated_ids is not None:
            state.generated_ids = generated_ids

        state.tokenizer = self.tokenizer

        for i in range(max_steps):
            if cache is None and i == 0 and generated_ids is not None:
                ids = torch.concat([state.prompt_ids, state.generated_ids], 1)
            else:
                ids = state.last_step_ids

            with torch.no_grad():
                output = self.model(
                    ids,
                    attention_mask=state.attention_mask,
                    cache=state.cache,
                    store_cache=True,
                    positions=state.positions,
                )

            state.step(
                cache=output.cache,
                predicted_ids=generation_step(logits_transform, output),
                stop_condition=stop_condition,
            )

            if not state.is_finished:
                yield False, state
            else:
                return True, state


class PartitionCache:
    def __init__(self, *cache, device="cpu"):
        self.cache = list(cache)
        self.device = device

    def add(self, batch_cache, index):
        cache = []
        for c in batch_cache:
            cache.append(KeyValueCache(c.key.to(self.device), c.value.to(self.device)))

        self.cache.append(cache)

        return self

    def pop(self, num: int, device="cuda"):
        if not self.cache:
            return None

        outputs = []

        i = 0
        while i < num:
            current = self.cache.pop()
            batch_size = current[0].key.size(0)

            if batch_size == 0:
                continue

            if i + batch_size <= num:
                outputs.append(
                    [
                        KeyValueCache(key=c.key.to(device), value=c.value.to(device))
                        for c in current
                    ]
                )
                i += batch_size
            else:
                left, right = [], []

                for c in current:
                    left.append(
                        KeyValueCache(
                            key=c.key[: num - i].to(device),
                            value=c.value[: num - i].to(device),
                        )
                    )
                    right.append(
                        KeyValueCache(key=c.key[num - i :], value=c.value[num - i :])
                    )
                i = num
                outputs.append(left)
                if right[0].key.size(0) > 0:
                    self.cache.append(right)

        if len(outputs) == 1:
            return outputs[0]

        output = []

        for i in range(len(outputs[0])):
            output.append(
                KeyValueCache(
                    key=pad_and_stack([o[i].key for o in outputs]),
                    value=pad_and_stack([o[i].value for o in outputs]),
                )
            )

        return output


class PartitionGenerator(StringGenerator):
    def __call__(self, df, config, batch_size=1024, max_steps=10):
        return self.generate(
            df, config=config, batch_size=batch_size, max_steps=max_steps
        )

    def generate(self, df, config, batch_size=1024, max_steps=10):
        """
        Generate text using the given prompts.

        :param prompts:
            Prompts to generate from.
        :param config:
            Generator configuraton.
        :returns:
            Strings generated for the prompts.
        """

        df["attention_mask"] = df["attention_mask"]

        completed_ids, completed_seq_ids = [], []
        self.inner.tokenizer = self.tokenizer

        num_samples = df["input_ids"].size
        i, n_tokens_generated, n_samples_done = 0, 0, 0
        it, n_batches = 0, 0

        df["index"] = df.index

        current_cache, next_cache = PartitionCache(), PartitionCache()

        with tqdm(total=num_samples, desc="Generating...", dynamic_ncols=True) as pbar:
            while n_samples_done < num_samples:
                results_list: List[Results] = []
                for batch in InMemoryLoader(df, batch_size=batch_size):
                    ids = batch["input_ids"]
                    _batch_size = ids.size(0)
                    attention_mask = AttentionMask(
                        batch["attention_mask"].to(torch.bool)
                    )

                    for generation_step in self.inner.generate_df(
                        ids=ids,
                        attention_mask=attention_mask,
                        config=config,
                        generated_ids=batch.get("generated_ids"),
                        index=batch["index"],
                        cache=current_cache.pop(_batch_size),
                        max_steps=max_steps,
                    ):
                        n_tokens_generated += _batch_size
                        _, state = generation_step
                        pbar.set_postfix(
                            {
                                "token": n_tokens_generated,
                                "batch": n_batches,
                                "it": it,
                            },
                            refresh=True,
                        )

                    results: Results = state.results()
                    if state.completed_ids is not None:
                        _num_samples_done = state.completed_ids.size(0)
                        pbar.update(_num_samples_done)
                        n_samples_done += _num_samples_done
                        completed_ids.append(results.completed_ids)
                        completed_seq_ids.append(results.completed_seq_ids)
                    if results.prompt_ids is not None:
                        results_list.append(results)

                    i += batch["input_ids"].size(0)
                    n_batches += 1

                    if state.cache is not None:
                        next_cache.add(state.cache, results.seq_ids)

                    pbar.set_postfix(
                        {
                            "token": n_tokens_generated,
                            "batch": n_batches,
                            "it": it,
                        },
                        refresh=True,
                    )

                if results_list:
                    df = {
                        "input_ids": pad_and_stack(
                            [r.prompt_ids for r in results_list]
                        ),
                        "generated_ids": pad_and_stack(
                            [r.generated_ids for r in results_list]
                        ),
                        "attention_mask": pad_and_stack(
                            [r.masks for r in results_list]
                        ),
                        "index": torch.concat([r.seq_ids for r in results_list]),
                    }
                it += 1
                pbar.set_postfix(
                    {
                        "token": n_tokens_generated,
                        "batch": n_batches,
                        "it": it,
                    },
                    refresh=True,
                )

                current_cache = next_cache
                next_cache = PartitionCache()

        tokens = pad_and_stack(completed_ids)
        text = self.tokenizer.decode(tokens.tolist())

        return text


def get_config(generator, config):
    eos_id = generator.default_config.eos_id if config.eos_id is None else config.eos_id
    max_generated_pieces = (
        generator.default_config.max_generated_pieces
        if config.max_generated_pieces is None
        else config.max_generated_pieces
    )
    config = dataclasses.replace(
        config, eos_id=eos_id, max_generated_pieces=max_generated_pieces
    )

    return config


@dataclass
class Results:
    completed_ids: torch.Tensor
    completed_seq_ids: torch.Tensor
    prompt_ids: Optional[torch.Tensor]
    generated_ids: Optional[torch.Tensor]
    seq_ids: Optional[torch.Tensor]
    masks: Optional[torch.Tensor]


class CuDFGeneratorState(Generic[CacheT]):
    """
    Stores the state of the generation process and tracks
    the sequences being generated.
    """

    attention_mask: AttentionMask
    cache: Optional[List[CacheT]]
    positions: torch.Tensor
    seq_ids: torch.Tensor
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor

    def __init__(
        self,
        *,
        attention_mask: AttentionMask,
        cache: Optional[List[CacheT]],
        prompt_ids: torch.Tensor,
        index=None,
    ) -> None:
        """
        Construct a generator state.

        :param attention_mask:
            Attention mask for the prompts.
        :param cache:
            Transformer model cache.
        :param prompt_ids:
            Batch of prompts.

            *Shape:* ``(batch_size, seq_len)``
        """
        device = prompt_ids.device
        assert (
            attention_mask.device == device
        ), f"Attention mask device '{attention_mask.device}' is not same as prompt ids device '{prompt_ids.device}'"
        self.attention_mask = attention_mask
        self.positions = attention_mask.bool_mask.int().cumsum(-1) - 1
        self.cache = cache

        if cache is not None:
            self.positions = self.positions.max(-1, keepdim=True).values + 1

        self.index = index
        self.seq_ids = torch.arange(0, self.attention_mask.shape[0], device=device)
        self.prompt_ids = prompt_ids
        self.generated_ids = torch.zeros(
            (prompt_ids.size(0), 0), dtype=prompt_ids.dtype, device=device
        )
        self.completed_ids, self.completed_seq_ids = None, None

    @property
    def is_finished(self):
        """
        Whether all sequences have finished generating.

        :returns:
            ``True`` iff all sequences have finished generating.
        """
        return len(self.seq_ids) == 0

    @property
    def last_step_ids(self) -> torch.Tensor:
        """
        Identifiers generated in the last step.

        :returns:
            Generated identifiers. Prompt identifiers are returned
            when the generator has not stepped yet.
        """
        if not self.generated_ids.size(1):
            return self.prompt_ids

        return self.generated_ids[:, -1:]

    def step(
        self,
        *,
        cache: List[CacheT],
        predicted_ids: torch.Tensor,
        stop_condition: StopCondition,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step the generation state.

        :param cache:
            Model cache from the last model call.
        :param generated_ids:
            Tensor containing generated IDs.

            *Shape:* ``(batch_size, 1)``
        :param stop_condition:
            Generation stop condition.
        :returns:
            Sequence identifiers and piece IDs.

            *Shape:* ``(batch_size), (batch_size, 1)``
        """
        # We update the state before removing completed sequences, so that
        # stopping conditions get a consistent view.
        self.cache = cache
        self.generated_ids = torch.concat([self.generated_ids, predicted_ids], 1)
        self.attention_mask = self.attention_mask.extend_length(
            count=1, fill_value=True
        )
        self.positions = self.positions.max(-1, keepdim=True).values + 1

        # Determine which sequences are done generating and remove them.
        completed_exclude = torch.zeros_like(predicted_ids, dtype=torch.bool)
        completed_include = torch.zeros_like(predicted_ids, dtype=torch.bool)
        stop_condition.update_completed(
            state=self,
            completed_exclude=completed_exclude,
            completed_include=completed_include,
        )

        self._remove_completed((completed_exclude ^ completed_include).view(-1))

        # return seq_ids, last_step_ids

    def _remove_completed(self, completed: torch.Tensor):
        """
        Remove completed sequences.

        :param completed:
            Tensor indicating for the active sequences whether they are completed.

        :meta private:
        """
        if torch.any(completed).item():
            if self.completed_ids is None:
                self.completed_ids = self.generated_ids[completed]
                self.completed_seq_ids = self.seq_ids[completed]
            else:
                self.completed_ids = pad_and_stack(
                    [self.completed_ids, self.generated_ids[completed]]
                )
                self.completed_seq_ids = torch.concat(
                    [self.completed_seq_ids, self.seq_ids[completed]]
                )

            not_completed = completed.logical_not()
            self.generated_ids = self.generated_ids[not_completed]
            self.attention_mask = self.attention_mask.filter_batch_items(not_completed)
            if self.cache is not None:
                self.cache = [
                    layer_cache.filter_batch_items(not_completed)
                    for layer_cache in self.cache
                ]
            self.prompt_ids = self.prompt_ids[not_completed]
            self.positions = self.positions[not_completed]
            self.seq_ids = self.seq_ids[not_completed]

    def get_ids(self, seq_ids):
        return self.index[seq_ids] if self.index is not None else seq_ids

    def results(self) -> Results:
        prompt_ids, generated_ids, seq_ids, masks, completed_seq_ids = (
            None,
            None,
            None,
            None,
            None,
        )
        if not self.is_finished:
            prompt_ids = self.prompt_ids
            if self.generated_ids.size(1):
                generated_ids = self.generated_ids
                masks = self.attention_mask.bool_mask.reshape(generated_ids.size(0), -1)
            seq_ids = self.get_ids(self.seq_ids)

        completed_ids = self.completed_ids
        if completed_ids is not None:
            completed_seq_ids = self.get_ids(self.completed_seq_ids)

        return Results(
            completed_ids=completed_ids,
            completed_seq_ids=completed_seq_ids,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            seq_ids=seq_ids,
            masks=masks,
        )


def pad_and_stack(tensor_list):
    # Find the maximum size along the last dimension
    max_size = max(tensor.shape[1] for tensor in tensor_list)

    # Initialize a list to store the padded tensors
    padded_tensors = []

    for tensor in tensor_list:
        # Calculate the amount of padding needed for this tensor
        padding_needed = max_size - tensor.shape[1]

        # Apply the padding
        padded_tensor = F.pad(tensor, pad=(0, padding_needed), mode="constant", value=0)

        # Add the padded tensor to the list
        padded_tensors.append(padded_tensor)

    return torch.concat(padded_tensors)
