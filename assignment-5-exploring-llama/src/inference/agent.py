from typing import List, Dict, Tuple, Optional, Union
from contextlib import contextmanager
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BatchLayout, PaddingSide, TruncateSide

from ..modeling.prompt import PromptType, PromptTemplate

from ..modeling.config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..utils import convert_to_list


class DecodeStrategy(Enum):
    """Decode Strategies Enum"""
    
    GREEDY = "greedy"
    SAMPLING = "sampling"


@config_dataclass
class InferenceConfig(BaseConfig):
    """Inference Configurations Dataclass"""
    
    # generation configurations
    decode_strategy: DecodeStrategy = DecodeStrategy.GREEDY
    temperature: float = 1.0
    max_new_tokens: int = make_required_field() # NOTE: we allow neither infinite generation nor early stopping for simplicity
    top_p: float = 1.0 # NOTE: only used when using sampling decode strategy
    top_k: int = 50 # NOTE: only used when using sampling decode strategy
    streaming: bool = False # NOTE: used when only one single user query is requested at a time, i.e. `inferred_batch_size == 1`
    sampling_seed: Optional[int] = None # NOTE: only used when using sampling decode strategy, if None then do not set seed
    
    # padding configurations
    batch_layout: BatchLayout = make_fixed_field(BatchLayout.STACK) # NOTE: we only allow stacking for simplicity
    padding_side: PaddingSide = PaddingSide.LEFT
    pad_to_multiple_of: int = 1
    
    # truncate configurations
    truncate_length: Optional[int] = None # NOTE: if None, then no truncation
    truncate_side: TruncateSide = TruncateSide.RIGHT
    
    # common configurations
    device: str = "cpu"

    def __post_init__(self):
        """Post-initialization method for InferenceConfig"""
        super().__post_init__()

        assert self.pad_to_multiple_of > 0 and (
            (self.pad_to_multiple_of & (self.pad_to_multiple_of - 1)) == 0
        ), "pad_to_multiple_of must be a power of 2"

        if self.truncate_length is not None and self.truncate_side == TruncateSide.MIDDLE:
            assert self.truncate_length % 2 == 0, "truncate_length must be even when truncate_side is MIDDLE"


class InferenceAgent(nn.Module):
    """Inference Agent module"""
    
    def __init__(
        self,
        config: InferenceConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
    ):
        """Initialize Inference Agent module
        
        Args:
            config (InferenceConfig): Inference Configurations
            model (BaseModel): the inner causal language model, which supports the common APIs of `BaseModel`
            tokenizer (BaseTokenizer): the inner tokenizer, which supports the common APIs of `BaseTokenizer`
        """
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self._prompts: Dict[PromptType, PromptTemplate] = {}
        
    def set_prompt(
        self,
        prompt_template: PromptTemplate,
        prompt_type: PromptType = PromptType.SYSTEM,
    ) -> None:
        """Set the prompt template
        
        Args:
            prompt_template (PromptTemplate): the prompt template
            prompt_type (PromptType): the prompt type
        """
        self._prompts[prompt_type] = prompt_template
            
    def get_prompt(
        self,
        prompt_type: PromptType = PromptType.SYSTEM
    ) -> PromptTemplate:
        """Get the prompt template
        
        Args:
            prompt_type (PromptType): the prompt type
        
        Returns:
            PromptTemplate: the prompt template
        """
        return self._prompts[prompt_type]
    
    def _truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        # truncation is disabled or not needed
        if self.config.truncate_length is None or input_ids.size(-1) <= self.config.truncate_length:
            return input_ids

        if self.config.truncate_side == TruncateSide.LEFT:
            return input_ids[:, -self.config.truncate_length:]
        elif self.config.truncate_side == TruncateSide.RIGHT:
            return input_ids[:, :self.config.truncate_length]
        elif self.config.truncate_side == TruncateSide.MIDDLE:
            half_length = self.config.truncate_length // 2
            return torch.cat([input_ids[:, :half_length], input_ids[:, -half_length:]], dim=-1)

    def _pad(self, input_ids_batch: List[torch.Tensor]) -> torch.Tensor:
        pad_token_id = self.tokenizer.eos_token_id if self.config.padding_side == PaddingSide.RIGHT else self.tokenizer.bos_token_id
        padding_value = pad_token_id

        max_len = max(len(ids) for ids in input_ids_batch)
        if self.config.pad_to_multiple_of > 1:
            max_len = ((max_len + self.config.pad_to_multiple_of - 1) // self.config.pad_to_multiple_of) * self.config.pad_to_multiple_of

        padded_batch = []
        for input_ids in input_ids_batch:
            pad_len = max_len - len(input_ids)
            if self.config.padding_side == PaddingSide.RIGHT:
                padding = [padding_value] * pad_len
                padded_input_ids = torch.cat([input_ids, torch.tensor(padding, dtype=input_ids.dtype)], dim=-1)
            else:
                padding = [padding_value] * pad_len
                padded_input_ids = torch.cat([torch.tensor(padding, dtype=input_ids.dtype), input_ids], dim=-1)
            padded_batch.append(padded_input_ids)
        return torch.stack(padded_batch)
    
    @torch.no_grad()
    def forward(
        self, 
        query: Union[str, List[str]], 
        **kwargs: Optional[Dict[str, str]]
    ) -> List[Dict[PromptType, str]]:
        """The forward pass of the Inference Agent module
        
        Args:
            query (Union[str, List[str]]): a single query prompt or a batch of user query prompts \
                as the core distinct instructions to ask the model to respond, \
                appended to the end of the complete prompt with the same system prompt and context prompt
                NOTE: when is a streaming mode, the query should be a single prompt
            kwargs (dict): additional keyword arguments to be passed to format the prefixed prompt templates
                NOTE: if certain key in `kwargs` are found in both system prompt template and context prompt template, \
                    the corresponding value will share in both of them as well
        Returns:
            List[Dict[PromptType, str]]: the list of dictionaries, \
                each of which should contain every prompt type in `PromptType` (key) and the corresponding prompt (value)
            NOTE: to simplify, we do not use early stopping strategy since the stopping point for each response might vary, \
                thus the length of the latent token ids for each response is ensured to be `max_new_tokens`
        """
        # set System and Context prompts
        system_prompt_template = self._prompts.get(PromptType.SYSTEM)
        context_prompt_template = self._prompts.get(PromptType.CONTEXT)
        system_prompt_str = system_prompt_template.forward(**kwargs) if system_prompt_template else ""
        context_prompt_str = context_prompt_template.forward(**kwargs) if context_prompt_template else ""

        queries = convert_to_list(query)
        results = []
        # iterate over each query
        for q in queries:
            prompt_dict = {}
            prompt_str = system_prompt_str + context_prompt_str + q

            prompt_dict[PromptType.SYSTEM] = system_prompt_str
            prompt_dict[PromptType.CONTEXT] = context_prompt_str
            prompt_dict[PromptType.QUERY] = q
            prompt_dict[PromptType.PROMPT] = prompt_str

            input_ids = self.tokenizer.encode(prompt_str)[0]  # a tensor
            truncated_input_ids = self._truncate(input_ids.unsqueeze(0)).squeeze(0)

            input_tensor = truncated_input_ids.unsqueeze(0).to(self.config.device)

            self.model.reset_kv_cache()
            generated_ids = []

            for _ in range(self.config.max_new_tokens):
                next_token_logits = self.model(input_tensor)

                if self.config.decode_strategy == DecodeStrategy.GREEDY:
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                elif self.config.decode_strategy == DecodeStrategy.SAMPLING:
                    if self.config.sampling_seed is not None:
                        torch.manual_seed(self.config.sampling_seed)

                    if self.config.temperature != 1.0:
                        next_token_logits = next_token_logits / self.config.temperature

                    if self.config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        top_p_mask = cumulative_probs < self.config.top_p
                        top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
                        sorted_logits[~top_p_mask] = float('-inf')
                        next_token_logits = next_token_logits.scatter(-1, sorted_indices, sorted_logits)
                    elif self.config.top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                        mask = torch.full_like(next_token_logits, float('-inf')).scatter(-1, top_k_indices, top_k_values)
                        next_token_logits = mask

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    raise ValueError(f"Invalid decode strategy: {self.config.decode_strategy}")

                generated_ids.append(next_token_id.item())
                input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0)], dim=-1)

            response_str = self.tokenizer.decode(torch.tensor(generated_ids))[0]

            prompt_dict[PromptType.RESPONSE] = response_str
            prompt_dict[PromptType.ALL] = prompt_str + response_str
            results.append(prompt_dict)

        return results
    
    @staticmethod
    def load_generation_config(
        config_file: str, 
        **extra_configs
    ) -> InferenceConfig:
        """Load config from the original original Llama generation config
        
        Args:
            config_file(str): path to the config file of the original original Llama generation config in .json format
            extra_configs(dict, optional): extra (key, value) config pair(s), to overwrite `config.key = value`, \
                helpful to set some configurations that are neither fixed nor provided in the original config such as `device`, `seed`, etc.
                NOTE: if any required configuration is not found in the original config, you are supposed to pass it in `extra_configs`, \
                    otherwise, a `ValueError` will be raised.
        Returns:
            InferenceConfig: an InferenceConfig object initialized from the config file
        """
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # required field
        if "max_new_tokens" not in config_dict and "max_new_tokens" not in extra_configs:
            raise ValueError("Required configuration 'max_new_tokens' not found.")
        
        # configuration fields to read from the config file
        readable_fields = ["temperature", "max_new_tokens", "top_p", "top_k", "streaming", "sampling_seed", "batch_layout", "padding_side", "pad_to_multiple_of", "truncate_length", "truncate_side"]
        
        loaded_config = {}
        for k, v in config_dict.items():
            if k in readable_fields:
                loaded_config[k] = v

        # handle decode strategy
        decode_strategy_str = config_dict.get("do_sample", False)
        loaded_config["decode_strategy"] = DecodeStrategy.SAMPLING if decode_strategy_str else DecodeStrategy.GREEDY

        # update with extra configurations
        loaded_config.update(extra_configs)

        return InferenceConfig(**loaded_config)