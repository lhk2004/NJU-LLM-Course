from typing import Optional, Tuple, Union, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

# from assignment2 implementations
from ..mlp import MLPActivationType

# from assignment3 implementations
from ..attention import AttnQKVLayout, AttnQKVPackFormat

# from assignment4 implementations
from ..transformer import (
    TransformerConfig,
    TransformerDecoderKVCache,
    TransformerDecoderBlock,
)

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ...utils import (
    convert_to_list,
    load_json,
    load_safetensors,
)

from .base import BaseTokenizer, BaseModel


@config_dataclass
class LlamaConfig(TransformerConfig):
    """Llama Configurations Dataclass
    NOTE: we will fix some configurations unavailable in the original Llama config or troublesome to handle with the pretrained params, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py#L26)
    """
    
    # fixed rope configurations
    rope_ratio: int = make_fixed_field(1)
    rope_dynamic: bool = make_fixed_field(False)
    
    # fixed normalization configurations
    group_size: Optional[int] = make_fixed_field(None)
    
    # fixed attention configurations
    qkv_pack_format: AttnQKVPackFormat = make_fixed_field(AttnQKVPackFormat.Q_K_V)
    qkv_layout: AttnQKVLayout = make_fixed_field(AttnQKVLayout.BSHD)
    window_size: Optional[int] = make_fixed_field(None)
    causal: bool = make_fixed_field(True)
    softmax_scale: Optional[float] = make_fixed_field(None)
    softmax_cap: Optional[float] = make_fixed_field(None)
    softmax_temp: float = make_fixed_field(1.0)
    softmax_clip_range: Tuple[float, float] = make_fixed_field((0., 1.))
    apply_qk_norm: bool = make_fixed_field(False)
    qk_norm_group_size: Optional[int] = make_fixed_field(None)
    
    # fixed sparse mlp configurations
    num_experts: Optional[int] = make_fixed_field(None)
    moe_topk: int = make_fixed_field(1)
    

class LlamaTokenizer(BaseTokenizer):
    """Llama Tokenizer module
    This is a tokenizer module that holds an instance of `Tokenizer` from `tokenizers` and loads its pretrained vocabulary and configuration, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/tokenization_llama.py#L54)
    with which this module implements the common APIs of `BaseTokenizer`
    """
    
    def __init__(
        self,
        vocab_file: str,
        config_file: str,
    ):
        """Initialize Llama Tokenizer module
        
        Args:
            vocab_file (str): path to the file with the pretrained vocabulary in .json format
            config_file (str): path to the file with the configuration in .json format
        """
        super().__init__()
        
        self.vocab_file = vocab_file
        self.config_file = config_file
        
        self._tokenizer = Tokenizer.from_file(vocab_file)
        
        self.config = load_json(config_file)
        
        self._bos_token = self.config["bos_token"]
        self._eos_token = self.config["eos_token"]
        self._pad_token = self.config.get("pad_token", self._eos_token)
        self._bos_token_id, self._eos_token_id, self._pad_token_id = [
            self._tokenizer.token_to_id(token)
            for token in [self._bos_token, self._eos_token, self._pad_token]
        ]
    
    def encode(
        self,
        prompt: Union[str, List[str]],
    ) -> List[torch.LongTensor]:
        """Encode a single or a list of prompt(s) into a list of tensors of encoded ids
        
        Args:
            prompt (Union[str, List[str]]): a single or a list of prompt(s) to be encoded
        
        Returns:
            List[torch.LongTensor]: a list of 1-dim tensors of encoded ids, each with shape: [seq_len,], \
                NOTE: if the input is a single prompt, the output will be a list with length 1
        """
        
        # convert to prompt list
        prompt = convert_to_list(prompt)
        
        # encode each prompt
        encoded_ids = [
            torch.tensor(
                encoded.ids,
                dtype=torch.long,
            )
            for encoded in self._tokenizer.encode_batch(prompt)
        ]
            
        return encoded_ids
    
    def decode(
        self,
        encoded_ids: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> List[str]:
        """Decode a single or a list of tensor(s) of encoded ids into a list of prompts
        
        Args:
            encoded_ids (Union[torch.LongTensor, List[torch.LongTensor]]): a single or a list of tensor(s) of encoded ids to be decoded, \
                NOTE: each tensor is a 1-dim tensor with shape: [seq_len,]
        
        Returns:
            List[str]: a list of prompt strings, each with length: `seq_len` (stripped the special tokens), \
                NOTE: if the input is a single tensor of encoded ids, the output will be a list with length 1
        """
        
        # convert to encoded ids list
        encoded_ids = convert_to_list(encoded_ids)
        encoded_ids = [encoded.tolist() for encoded in encoded_ids]
        
        # decode each encoded ids
        return self._tokenizer.decode_batch(encoded_ids)
    
    @property
    def vocab_size(self) -> int:
        """The vocab size of the tokenizer"""
        return self._tokenizer.get_vocab_size()
    
    @property
    def pad_token(self) -> str:
        """The pad token of the tokenizer"""
        return self._pad_token
    
    @property
    def pad_id(self) -> int:
        """The pad token id of the tokenizer"""
        return self._pad_token_id
    
    @property
    def bos_token(self) -> str:
        """The bos token of the tokenizer"""
        return self._bos_token
    
    @property
    def bos_id(self) -> int:
        """The bos token id of the tokenizer"""
        return self._bos_token_id
    
    @property
    def eos_token(self) -> str:
        """The eos token of the tokenizer"""
        return self._eos_token
    
    @property
    def eos_id(self) -> int:
        """The eos token id of the tokenizer"""
        return self._eos_token_id


class LlamaModel(BaseModel):
    """Llama Model module
    This is a variant modeling of the `LlamaForCausalLM` compatible with the pretrained parameters, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L1105)
    which supports the common APIs of `BaseModel` and the extra loading methods \
        to load configurations and pretrained weights from the original HF repo
    """
    
    def __init__(
        self,
        config: LlamaConfig,
    ):
        """Initialize Llama Model module
        
        Args:
            config (LlamaConfig): Llama configuration
        """
        super().__init__()
        self.config = config
        self.block = TransformerDecoderBlock(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """The forward pass of the Llama Model module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, either as a training sample or as a query prompt, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
            labels(Optional[torch.LongTensor]): the vocab ids for the training labels, with the same shape as `input_ids` (and the `ignore_index` is set to `-100` as default by pytorch)
            NOTE: due to the nll loss kernel restriction, the dtype of labels should be exactly int64 (torch.long) if provided
            temperature (float, optional): temperature for softmax during inference, default to 1.0, in the range of (0., +inf)
        Returns:
            torch.Tensor: output tensor, whose value depends on the model's state:
                1. for training, the value is just the cross-entroy scalar loss by `mean` reduction between the vocab logits and the given labels
                2. for inference, the value is the next-token probability distribution for every sequence in the batch, with shape: [inferred_batch_size, vocab_size]
            NOTE: the `inferred_batch_size` is inferred from `cu_seqlens` if provided (i.e. `inner_batch_size`), otherwise from `input_ids` (i.e. `batch_size`)
        """
        if cu_seqlens is not None:
            assert input_ids.shape[0] == 1, f"Expected input batch size to be 1 when `cu_seqlens` is enabled, but got: {input_ids.shape[0]}"

        logits = self.block(input_ids, cu_seqlens=cu_seqlens)

        if labels is not None:
            # training mode
            assert labels.dtype == torch.int64, f"Expected labels dtype to be int64, but got: {labels.dtype}"
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return loss
        else:
            # evaluation mode
            with torch.no_grad():
                scaled_logits = logits[:, -1, :] / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                return probs
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        return self.block.get_kv_cache()
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        self.block.set_kv_cache(kv_cache)
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        self.block.reset_kv_cache()
    
    def reset_parameters(self):
        """Initialize learnable parameters for Llama Model module"""
        self.block.reset_parameters()
    
    def load_parameters(self, params_files: Union[str, List[str]]) -> None:
        """Load pretrained params from the original LlamaForCausalLM
        
        Args:
            params_files(Union[str, List[str]]): path to the pretrained params file(s) of the original LlamaForCausalLM in .safetensors format
                NOTE: if there're multiple params files, it is ensured that there is NO param name conflict
        """
        state_dict = load_safetensors(params_files)
       
        def map_param_name(name: str) -> str:
            if name == "model.embed_tokens.weight":
                return "block.vocab_embedding.embedding_table"
            elif name.startswith("model.layers"):
                parts = name.split(".")
                block_index = int(parts[2])
                mapped_name = f"block.layers.{block_index}."
                if parts[3] == "self_attn":
                    if parts[4] in {"q_proj", "k_proj", "v_proj"}:
                        return None  # handled afterwards
                    elif parts[4] == "o_proj":
                        return mapped_name + "o_proj"
                elif parts[3] == "mlp":
                    if parts[4] == "gate_proj":
                        return mapped_name + "mlp.W_gate"
                    elif parts[4] == "up_proj":
                        return mapped_name + "mlp.W_up"
                    elif parts[4] == "down_proj":
                        return mapped_name + "mlp.W_down"
                elif parts[3] == "input_layernorm":
                    return mapped_name + "attn_norm.gamma"
                elif parts[3] == "post_attention_layernorm":
                    return mapped_name + "mlp_norm.gamma"
                else:
                    raise ValueError(f"Unexpected parameter name: {name}")
            elif name == "model.norm.weight":
                return "block.final_norm.gamma"
            elif name == "model.lm_head.weight":
                return "block.lm_head.weight"
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        # collect q_proj, k_proj, v_proj weights
        qkv_weights = {}
        for name, param in state_dict.items():
            if "self_attn.q_proj.weight" in name or "self_attn.k_proj.weight" in name or "self_attn.v_proj.weight" in name:
                parts = name.split(".")
                block_index = int(parts[2])
                if block_index not in qkv_weights:
                    qkv_weights[block_index] = {"q_proj": None, "k_proj": None, "v_proj": None}
                if "q_proj.weight" in name:
                    qkv_weights[block_index]["q_proj"] = param
                elif "k_proj.weight" in name:
                    qkv_weights[block_index]["k_proj"] = param
                elif "v_proj.weight" in name:
                    qkv_weights[block_index]["v_proj"] = param

        # map parameter names
        mapped_state_dict = {}
        for k, v in state_dict.items():
            mapped_name = map_param_name(k)
            if mapped_name: # ignore qkv_proj
                if "W_gate" in mapped_name or "W_up" in mapped_name or "W_down" in mapped_name or "o_proj" in mapped_name:
                    mapped_state_dict[mapped_name] = v.transpose(0, 1)
                else:
                    mapped_state_dict[mapped_name] = v

        # handle qkv_proj
        for block_index, weights in qkv_weights.items():
            q_proj = weights["q_proj"]
            k_proj = weights["k_proj"]
            v_proj = weights["v_proj"]
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(f"Incomplete self-attn weights for block {block_index}")
            qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=0).transpose(0, 1)
            mapped_state_dict[f"block.layers.{block_index}.qkv_proj"] = qkv_proj

        # loading parameters
        self.load_state_dict(mapped_state_dict, strict=False)
    
    @staticmethod
    def load_config(config_file: str, **extra_configs) -> LlamaConfig:
        """Load config from the original original Llama config
        
        Args:
            config_file(str): path to the config file of the original original Llama config in .json format
            extra_configs(dict, optional): extra (key, value) config pair(s), to overwrite `config.key = value`, \
                helpful to set some configurations that are neither fixed nor provided in the original config such as `device`, `seed`, etc.
                NOTE: if any required configuration is not found in the original config, you are supposed to pass it in `extra_configs`, \
                    otherwise, a `ValueError` will be raised.
        Returns:
            LlamaConfig: a LlamaConfig object initialized from the config file
        """
        config_dict = load_json(config_file)
        config_updates = {}
        
        # required fields
        config_updates["num_layers"] = config_dict.get("num_hidden_layers", None)
        config_updates["hidden_size"] = config_dict.get("hidden_size", None)
        config_updates["ffh_size"] = config_dict.get("intermediate_size", None)
        config_updates["max_seq_len"] = config_dict.get("max_position_embeddings", None)
        config_updates["max_seq_len"] = config_dict.get("rope_scaling", {}).get("original_max_position_embeddings", None)
        config_updates["vocab_size"] = config_dict.get("vocab_size", None)
        config_updates["head_dim"] = config_dict.get("head_dim", None)
        config_updates["num_q_head"] = config_dict.get("num_attention_heads", None)
        config_updates["num_kv_head"] = config_dict.get("num_key_value_heads", None)

        # optional fields
        if "attention_dropout" in config_dict:
            config_updates["softmax_dropout_rate"] = config_dict["attention_dropout"]
        if "hidden_act" in config_dict:
            config_updates["activation_type"] = MLPActivationType(config_dict["hidden_act"])
        if "rms_norm_eps" in config_dict:
            config_updates["eps"] = config_dict["rms_norm_eps"]
        if "tie_word_embeddings" in config_dict:
            config_updates["lm_head_tied"] = config_dict["tie_word_embeddings"]
        if "torch_dtype" in config_dict:
            config_updates["param_dtype"] = getattr(torch, config_dict['torch_dtype'])
        if "rope_theta" in config_dict:
            config_updates["rope_base"] = config_dict["rope_theta"]

        # extra configs to overwrite current configs
        config_updates.update(extra_configs)

        # check for required fields and raise ValueError if missing
        for field_name in LlamaConfig.__dataclass_fields__:
            field = LlamaConfig.__dataclass_fields__[field_name]
            if field.metadata.get("required") and (field_name not in config_updates or config_updates[field_name] is None):
                raise ValueError(f"Required config field '{field_name}' is missing.")

        return LlamaConfig(**config_updates)
    
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        return self.block.num_parameters(learnable_only=learnable_only, unit=unit)
    
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        return self.block.num_memory_footprint(unit=unit)