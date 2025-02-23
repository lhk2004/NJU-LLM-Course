from typing import Optional, Tuple, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

from .config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

@config_dataclass
class TransformerConfig(BaseConfig):
    """Transformer Configurations Dataclass"""
    
    # common transformer configurations
    num_layers: int = make_required_field()
    hidden_size: int = make_required_field()
    ffh_size: int = make_required_field()
    max_seq_len: int = make_required_field()
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = make_fixed_field(0)
    world_size: int = make_fixed_field(1)
    process_group: Optional[ProcessGroup] = make_fixed_field(None)
    
    # vocab embedding configurations
    vocab_size: int = make_required_field()
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = make_required_field()
    num_q_head: int = make_required_field()
    num_kv_head: int = make_required_field()
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        super().__init__()
        self.qkv_layout = qkv_layout
        self.num_layers = num_layers
        
        # initialize a dictionary to store the key-value tensors for each layer
        self.kv_cache = {layer_idx: None for layer_idx in range(num_layers)}

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """
        return self.kv_cache[layer_idx] is not None

    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        if self.kv_cache[layer_idx] is None:
            raise KeyError(f"No cache found for layer {layer_idx}")
        
        k, v, cu_seqlens = self.kv_cache[layer_idx]
        
        return k, v, cu_seqlens

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        if self.qkv_layout == AttnQKVLayout.THD and cu_seqlens is None:
            raise ValueError("The cu_seqlens must be provided if the qkv_layout is AttnQKVLayout.THD")
        
        self.kv_cache[layer_idx] = (k, v, cu_seqlens)

    def concat_kv_with_cu_seqlens(self, prev_kv, cur_kv, prev_cu_seqlens, cu_seqlens):
        '''Concatenate the keys or values with their cumulative sequence lengths along the sequence dimension'''
        new_kv = []

        for i in range(len(prev_cu_seqlens) - 1):
            prev_start = prev_cu_seqlens[i]
            prev_end = prev_cu_seqlens[i + 1]
            
            cur_start = cu_seqlens[i]
            cur_end = cu_seqlens[i + 1]

            new_kv.append(torch.cat([prev_kv[prev_start:prev_end], cur_kv[cur_start:cur_end]], dim=0))

        return torch.cat(new_kv, dim=0)

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        if self.qkv_layout == AttnQKVLayout.THD and cu_seqlens is None:
            raise ValueError("The cu_seqlens must be provided if the qkv_layout is AttnQKVLayout.THD")
        
        # if no cache exists, set the current keys and values
        if self.kv_cache[layer_idx] is None:
            self.set(layer_idx, k, v, cu_seqlens)
        # otherwise, append the new keys and values along the sequence dimension
        else:
            prev_k, prev_v, prev_cu_seqlens = self.kv_cache[layer_idx]

            if self.qkv_layout == AttnQKVLayout.BSHD:
                new_k = torch.cat([prev_k, k], dim=1)
                new_v = torch.cat([prev_v, v], dim=1)
            elif self.qkv_layout == AttnQKVLayout.SBHD:
                new_k = torch.cat([prev_k, k], dim=0)
                new_v = torch.cat([prev_v, v], dim=0)
            else:
                new_k = self.concat_kv_with_cu_seqlens(prev_k, k, prev_cu_seqlens, cu_seqlens)
                new_v = self.concat_kv_with_cu_seqlens(prev_v, v, prev_cu_seqlens, cu_seqlens)
            
            # optionally concatenate cumulative sequence lengths if provided
            if cu_seqlens is not None:
                assert self.qkv_layout == AttnQKVLayout.THD, "The cu_seqlens must be provided if the qkv_layout is AttnQKVLayout.THD"
                # e.g. prev_cu_seqlens: [0, 1, 2, 5] denotes 3 sequences with lengths [1, 1, 3] respectively,
                # and cu_seqlens: [0, 1, 2, 3] denotes 3 sequences with lengths [1, 1, 1] respectively,
                # so new_cu_seqlens should be [0, 2, 4, 8] denoting 3 sequences with lengths [2, 2, 4] respectively
                new_cu_seqlens = prev_cu_seqlens + cu_seqlens
            else:
                new_cu_seqlens = prev_cu_seqlens  # keep the same if no new cu_seqlens provided
            
            # update the cache with the newly concatenated tensors
            self.set(layer_idx, new_k, new_v, new_cu_seqlens)
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        self.kv_cache = {layer_idx: None for layer_idx in range(self.num_layers)}
 

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.param_dtype = config.param_dtype
        self.param_device = config.param_device

        # 1. self-attention layer initialization

        # GroupRMSNorm instead of layernorm
        self.attn_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 1,
            dtype=config.param_dtype,
            device=config.param_device
        )

        # QKV projection
        self.qkv_proj = nn.Parameter(torch.empty(
            config.hidden_size,
            config.num_q_head * config.head_dim + 2 * config.num_kv_head * config.head_dim,
            device=config.param_device,
            dtype=config.param_dtype))
        
        # positional embeddings
        self.rope = NTKAwareRoPE(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            ratio=config.rope_ratio,
            dynamic=config.rope_dynamic,
            device=config.param_device,
            dtype=config.param_dtype
        )

        # self-attention mechanism
        if config.online_attn_block_size is not None:
            self.attn = OnlineSlidingWindowAttn(
                seqlen_q=config.max_seq_len,
                seqlen_kv=config.max_seq_len,
                block_size_q=config.online_attn_block_size,
                block_size_kv=config.online_attn_block_size,
                head_dim=config.head_dim,
                num_q_head=config.num_q_head,
                num_kv_head=config.num_kv_head,
                window_size=config.window_size,
                causal=config.causal,
                softmax_scale=config.softmax_scale,
                softmax_cap=config.softmax_cap,
                softmax_temp=config.softmax_temp,
                apply_qk_norm=config.apply_qk_norm,
                group_size=config.qk_norm_group_size,
                eps=config.eps,
                init_range=config.norm_init_range,
                init_seed=config.init_base_seed + layer_idx + 2,
                device=config.param_device,
                dtype=config.param_dtype
            )
        else:
            self.attn = OfflineSlidingWindowAttn(
                head_dim=config.head_dim,
                num_q_head=config.num_q_head,
                num_kv_head=config.num_kv_head,
                qkv_pack_format=config.qkv_pack_format,
                qkv_layout=config.qkv_layout,
                window_size=config.window_size,
                causal=config.causal,
                softmax_dropout_rate=config.softmax_dropout_rate,
                softmax_dropout_seed=config.softmax_dropout_seed + layer_idx,
                softmax_scale=config.softmax_scale,
                softmax_cap=config.softmax_cap,
                softmax_temp=config.softmax_temp,
                softmax_clip_range=config.softmax_clip_range,
                apply_qk_norm=config.apply_qk_norm,
                group_size=config.qk_norm_group_size,
                eps=config.eps,
                init_range=config.norm_init_range,
                init_seed=config.init_base_seed + layer_idx + 2,
                device=config.param_device,
                dtype=config.param_dtype
            )

        # output projection
        self.o_proj = nn.Parameter(torch.empty(
            config.num_q_head * config.head_dim,
            config.hidden_size,
            device=config.param_device,
            dtype=config.param_dtype))

        # 2. MLP layer initialization

        # GroupRMSNorm instead of layernorm
        self.mlp_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 3,
            dtype=config.param_dtype,
            device=config.param_device
        )
        
        # MLP
        if config.num_experts is None:
            self.mlp = DenseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                init_base_seed=config.init_base_seed + layer_idx + 4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                device=config.param_device,
                dtype=config.param_dtype
            )
        else:
            self.mlp = SparseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                num_experts=config.num_experts,
                moe_topk=config.moe_topk,
                rank=self.config.rank,
                world_size=self.config.world_size,
                process_group=self.config.process_group,
                init_mean=config.gate_init_mean,
                init_std=config.gate_init_std,
                init_base_seed=config.init_base_seed + layer_idx + 4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                device=config.param_device,
                dtype=config.param_dtype
            )

        # initialize all learnable parameters
        self.reset_parameters()
    
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        qkv_layout = self.config.qkv_layout
        qkv_pack_format = self.config.qkv_pack_format
        num_q_head = self.config.num_q_head
        num_kv_head = self.config.num_kv_head
        head_dim = self.config.head_dim
        dtype = input.dtype
        device = input.device

        # check qkv_layout consistency
        if kv_cache is not None:
            assert qkv_layout == kv_cache.qkv_layout, "The qkv_layout of the layer should be consistent with the kv_cache"
        
        # convert input to the same dtype and device as the learnable parameters
        input = input.to(self.param_dtype).to(self.param_device)

        # R = X
        residual = input # [b, s, h]

        # X_tilda = Attn_norm(X)
        normalized_input = self.attn_norm(input)

        # qkv = X_tilda × W_QKV
        # Q, K, V = split(qkv)
        qkv = torch.matmul(normalized_input, self.qkv_proj) # [b, s, nq * hd + 2 * nkv * hd]
        q, k, v = torch.split(qkv, [num_q_head * head_dim, num_kv_head * head_dim, num_kv_head * head_dim], dim=-1)
        q = q.reshape(q.size(0), q.size(1), num_q_head, head_dim)    # [b, s, nq, hd]
        k = k.reshape(k.size(0), k.size(1), num_kv_head, head_dim)   # [b, s, nkv, hd]
        v = v.reshape(v.size(0), v.size(1), num_kv_head, head_dim)   # [b, s, nkv, hd]

        # handle QKV layout
        if self.config.qkv_layout == AttnQKVLayout.THD:
            assert cu_seqlens is not None, "cu_seqlens must be provided if qkv_layout is AttnQKVLayout.THD"
            # batch_size is ensured to be 1
            q = q.squeeze(0) # [s, nq, hd]
            k = k.squeeze(0) # [s, nkv, hd]
            v = v.squeeze(0) # [s, nkv, hd]
            cu_seqlens_q = cu_seqlens
            cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = None
            cu_seqlens_k = None
            if qkv_layout == AttnQKVLayout.BSHD:
                # q, k and v are already in [b, s, nh, hd]
                pass
            else: # qkv_layout == AttnQKVLayout.SBHD:
                q = q.transpose(0, 1) # [s, b, nq, hd]
                k = k.transpose(0, 1) # [s, b, nkv, hd]
                v = v.transpose(0, 1) # [s, b, nkv, hd]
        
        # find the offset of q in the original sequence (since it might only be a single token)
        # offset := the sequence length of k_cache
        if kv_cache is not None and kv_cache.has(self.layer_idx):
            k_cache, v_cache, past_cu_seqlens = kv_cache.get(self.layer_idx)
            if past_cu_seqlens is not None:
                offset = past_cu_seqlens[-1].item() if len(past_cu_seqlens) > 0 else 0
            else:
                if kv_cache.qkv_layout == AttnQKVLayout.BSHD:
                    offset = k_cache.shape[1]
                elif kv_cache.qkv_layout == AttnQKVLayout.SBHD:
                    offset = k_cache.shape[0]
                else:
                    raise ValueError("The cu_seqlens must be provided if the qkv_layout is AttnQKVLayout.THD")
        else:
            offset = 0

        # add positional embeddings
        if qkv_layout == AttnQKVLayout.BSHD:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        elif qkv_layout == AttnQKVLayout.SBHD:
            q = self.rope(q.transpose(0, 1), offset=offset).transpose(0, 1)
            k = self.rope(k.transpose(0, 1), offset=offset).transpose(0, 1)
        else: # qkv_layout == AttnQKVLayout.THD:
            if kv_cache is not None and kv_cache.has(self.layer_idx):
                _, _, prev_cu_seqlens = kv_cache.get(self.layer_idx)

            q = q.clone()
            k = k.clone()
            # cu_seqlens_q.size(0) - 1 is the batch size for q (also the batch size for k)
            for i in range(cu_seqlens_q.size(0) - 1):
                # offset is the index of the first token of this batch in the entire sequence
                if kv_cache is not None and kv_cache.has(self.layer_idx):
                    offset = prev_cu_seqlens[i+1].item() - prev_cu_seqlens[i].item()
                # extract the i-th batch
                input_q = q[cu_seqlens_q[i].item() : cu_seqlens_q[i+1].item(), :, :].unsqueeze(0) # [1, s, nq, hd]       
                input_k = k[cu_seqlens_k[i].item() : cu_seqlens_k[i+1].item(), :, :].unsqueeze(0) # [1, s, nkv, hd]
                # apply RoPE
                q[cu_seqlens_q[i].item() : cu_seqlens_q[i+1].item(), :, :] = self.rope(input_q, offset=offset).squeeze(0)
                k[cu_seqlens_k[i].item() : cu_seqlens_k[i+1].item(), :, :] = self.rope(input_k, offset=offset).squeeze(0)

        # update kv cache, then use the updated kv cache as new k and v
        if kv_cache is not None:
            if kv_cache.has(self.layer_idx):
                kv_cache.append(self.layer_idx, k, v, cu_seqlens)
                k, v, cu_seqlens_k = kv_cache.get(self.layer_idx)
            else:
                kv_cache.set(self.layer_idx, k, v, cu_seqlens)

        # handle QKV packing format
        if qkv_pack_format == AttnQKVPackFormat.Q_K_V:
            # Q, K, V are already separate tensors
            pass
        elif qkv_pack_format == AttnQKVPackFormat.Q_KV:
            # K and V are packed together along the num_head dimension
            k = torch.cat((k, v), dim=-2)
            v = None
        else: # self.qkv_pack_format == AttnQKVPackFormat.QKV:
            # Q, K and V are all packed together along the num_head dimension
            # which means other dimensions of Q, K and V have to be the same
            q = torch.cat((q, k, v), dim=-2)
            k = None
            v = None

        # self-attention
        if self.config.online_attn_block_size is None:
            # OfflineSlidingWindowAttn
            attn_output = self.attn(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k
                )
        else:
            # OnlineSlidingWindowAttn
            block_size_q = self.config.online_attn_block_size
            block_size_kv = self.config.online_attn_block_size
            num_q_head = self.config.num_q_head
            head_dim = self.config.head_dim
            # qkv_layout is ensured to be BSHD and qkv_pack_format is ensured to be Q_K_V
            # when applying OnlineSlidingWindowAttn
            # also, seqlen_q and seq_len_kv are ensured to be the same as max_seq_len
            batch_size, seqlen_q, _, _ = q.shape
            _, seqlen_kv, _, _ = k.shape

            num_block_q = (seqlen_q + block_size_q - 1) // block_size_q
            num_block_kv = (seqlen_kv + block_size_kv - 1) // block_size_kv

            global_o = torch.zeros((batch_size, seqlen_q, num_q_head, head_dim), dtype=self.param_dtype, device=self.param_device)
            global_lse = torch.full((batch_size, num_q_head, seqlen_q), fill_value=-torch.inf, dtype=torch.float32, device=self.param_device)

            for block_idx_q in range(num_block_q):
                start_q = block_idx_q * block_size_q
                end_q = min((block_idx_q + 1) * block_size_q, seqlen_q)

                q_block = q[:, start_q : end_q, :, :]

                for block_idx_kv in range(num_block_kv):
                    start_kv = block_idx_kv * block_size_kv
                    end_kv = min((block_idx_kv + 1) * block_size_kv, seqlen_kv)

                    k_block = k[:, start_kv : end_kv, :, :]
                    v_block = v[:, start_kv : end_kv, :, :]

                    self.attn.forward(
                        q=q_block,
                        k=k_block,
                        v=v_block,
                        global_o=global_o,
                        global_lse=global_lse,
                        block_idx_q=block_idx_q,
                        block_idx_kv=block_idx_kv,
                    )

            attn_output = global_o

        # [..., nq, hd] -> [..., nq * hd] (for the use of projection)
        if qkv_layout == AttnQKVLayout.BSHD or qkv_layout == AttnQKVLayout.SBHD:
            attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1)
        else: # qkv_layout == AttnQKVLayout.THD:
            attn_output = attn_output.reshape(attn_output.size(0), -1)

        # output projection
        attn_output_proj = torch.matmul(attn_output, self.o_proj)

        # reshape attn_output_proj to the same shape as residual
        # shape of q -> [b, s, hidden_size]
        if qkv_layout == AttnQKVLayout.BSHD:
            pass
        elif qkv_layout == AttnQKVLayout.SBHD:
            attn_output_proj = attn_output_proj.transpose(0, 1)
        else: # qkv_layout == AttnQKVLayout.THD:
            attn_output_proj = attn_output_proj.unsqueeze(0)

        # residual connection
        hidden_states = residual + attn_output_proj

        # MLP
        residual = hidden_states
        normalized_hidden_states = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states.to(dtype=dtype).to(device=device)
    
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Layer module"""
        # QKV projection
        torch.manual_seed(self.config.proj_init_seed + self.layer_idx + 1)
        nn.init.normal_(self.qkv_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        # output projection
        torch.manual_seed(self.config.proj_init_seed + self.layer_idx + 2)
        nn.init.normal_(self.o_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.param_dtype = config.param_dtype
        self.param_device = config.param_device
        self.lm_head_tied = config.lm_head_tied

        # input embedding layer
        self.vocab_embedding = ParallelVocabEmbedding(
            vocab_size=config.vocab_size,
            emb_size=config.hidden_size,
            rank=config.rank,
            world_size=config.world_size,
            process_group=config.process_group,
            init_mean=config.vocab_init_mean,
            init_std=config.vocab_init_std,
            init_base_seed=config.init_base_seed,
            device=config.param_device,
            dtype=config.param_dtype
        )

        # stacked decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # final normalization layer
        self.final_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )

        # output embedding layer (LM head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=config.param_device, dtype=config.param_dtype)

        # KV cache
        self.kv_cache = TransformerDecoderKVCache(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers
        )

        self.reset_parameters()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        input_device = input_ids.device
        input_ids = input_ids.to(device=self.param_device)

        # input embedding
        hidden_states = self.vocab_embedding(input_ids)

        # decoder layers
        for i, layer in enumerate(self.layers):
            if not self.training:
                hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens, kv_cache=self.kv_cache)
            else:
                hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens, kv_cache=None)

        # final normalization
        normalized_hidden_states = self.final_norm(hidden_states)

        # LM head
        logits = self.lm_head(normalized_hidden_states)

        return logits.to(device=input_device)
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        return self.kv_cache
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        self.kv_cache = kv_cache
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        self.kv_cache.reset()
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        # input embedding
        self.vocab_embedding.reset_parameters()

        # decoder layers
        for layer in self.layers:
            layer.reset_parameters()

        # final normalization
        self.final_norm.reset_parameters()

        # LM head
        if not self.lm_head_tied:
            torch.manual_seed(self.config.proj_init_seed)
            nn.init.normal_(self.lm_head.weight, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        else:
            self.lm_head.weight = self.vocab_embedding.embedding_table
     
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
        num_params = 0.0
        for p in self.parameters():
            if learnable_only:
                if p.requires_grad:
                    num_params += p.numel()
            else:
                num_params += p.numel()

        if unit == "K":
            num_params /= 1_000
        elif unit == "M":
            num_params /= 1_000_000
        elif unit == "B":
            num_params /= 1_000_000_000
        elif unit != "1":
            raise ValueError(f"Invalid unit: {unit}. Choose from '1', 'K', 'M', 'B'.")

        return num_params
    
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
        memory_footprint = 0.0

        for p in self.parameters():
            memory_footprint += p.numel() * p.element_size()

        if unit == "KB":
            memory_footprint /= 1_024
        elif unit == "MB":
            memory_footprint /= 1_024 ** 2
        elif unit == "GB":
            memory_footprint /= 1_024 ** 3
        elif unit != "B":
            raise ValueError(f"Invalid unit: {unit}. Choose from 'B', 'KB', 'MB', 'GB'.")

        return memory_footprint