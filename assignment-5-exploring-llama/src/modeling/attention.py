from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        self.window_size = window_size
        self.causal = causal
        self.softmax_dropout_rate = softmax_dropout_rate
        self.softmax_dropout_seed = softmax_dropout_seed
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(head_dim))
        self.softmax_cap = softmax_cap
        self.softmax_temp = softmax_temp
        self.softmax_clip_range = softmax_clip_range
        self.apply_qk_norm = apply_qk_norm
        self.group_size = group_size or head_dim
        self.eps = eps
        self.dtype = dtype
        self.device = device

        # apply Group RMSNorm if qk_norm is enabled
        if self.apply_qk_norm:
            self.q_norm = GroupRMSNorm(
                hidden_size=num_q_head * head_dim,
                group_size=self.group_size,
                eps=self.eps,
                init_range=init_range,
                init_seed=init_seed,
                dtype=dtype,
                device=device
            )
            self.k_norm = GroupRMSNorm(
                hidden_size=num_kv_head * head_dim,
                group_size=self.group_size,
                eps=self.eps,
                init_range=init_range,
                init_seed=init_seed,
                dtype=dtype,
                device=device
            )
    
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        # handle packing formats
        if self.qkv_pack_format == AttnQKVPackFormat.Q_K_V:
            # Q, K, V are separate tensors
            q_states = q
            k_states = k
            v_states = v
        elif self.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            # K and V are packed together along the num_head dimension
            q_states = q
            k_states, v_states = torch.split(k, [self.num_kv_head, self.num_kv_head], dim=-2)
        elif self.qkv_pack_format == AttnQKVPackFormat.QKV:
            # Q, K and V are all packed together along the num_head dimension
            # which means other dimensions of Q, K and V have to be the same
            q_states, k_states, v_states = torch.split(q, [self.num_q_head, self.num_kv_head, self.num_kv_head], dim=-2)
        
        # handle the layout (make sure the shape of Q, K and V is [batch_size, seq_len, num_head, head_dim])
        if self.qkv_layout == AttnQKVLayout.BSHD:
            pass
        elif self.qkv_layout == AttnQKVLayout.SBHD:
            q_states = q_states.transpose(0, 1)
            k_states = k_states.transpose(0, 1)
            v_states = v_states.transpose(0, 1)
        elif self.qkv_layout == AttnQKVLayout.THD:
            # extract a list of tensors from the original tensor, given cu_seqlens
            # element in the list: [1, var_seq_len, nh, hd] (i.e. batch_size = 1)
            q_states = self.extract_tensors(original_tensor=q_states, cu_seqlens=cu_seqlens_q)
            k_states = self.extract_tensors(original_tensor=k_states, cu_seqlens=cu_seqlens_k)
            v_states = self.extract_tensors(original_tensor=v_states, cu_seqlens=cu_seqlens_k)

        # handle variable sequence length
        if self.qkv_layout == AttnQKVLayout.THD:

            # q, k and v have the same batch size
            batch_size = len(q_states)

            # iterate over each batch (which makes q_i, k_i and v_i the shape of [1, var_seq_len, nh, hd])
            output_list = []
            for i in range(batch_size):
                q_states_i = q_states[i].unsqueeze(0)
                k_states_i = k_states[i].unsqueeze(0)
                v_states_i = v_states[i].unsqueeze(0)
                
                o = self.compute_attention_output(q_states_i, k_states_i, v_states_i)
                o = o.squeeze(0) # [1, seq_len_q, nh, hd] -> [seq_len_q, nh, hd]
                output_list.append(o)

            # concatenate outputs [seq_len_q, nh, hd] into [batch_size, seq_len_q, nh, hd]
            output = torch.cat(output_list, dim=0)
        
        # fixed sequence length in each batch
        else:
            output = self.compute_attention_output(q_states, k_states, v_states)

        # output should be transformed back to original layout
        if self.qkv_layout == AttnQKVLayout.BSHD:
            pass # output already in [b, s, nh, hd]
        elif self.qkv_layout == AttnQKVLayout.SBHD:
            output = output.transpose(0, 1)
        elif self.qkv_layout == AttnQKVLayout.THD:
            pass # output already in [total_s, nh, hd]

        return output.to(q.dtype).to(q.device)
    
    def compute_attention_output(self, q_states: torch.Tensor, k_states: torch.Tensor, v_states: torch.Tensor) -> torch.Tensor:
        """Compute attention output given q, k and v,
        where q in shape [b, sq, nh, hd],
        k and v in shape [b, skv, nh, hd],
        and return output in shape [b, sq, nh, hd]
        """
        # apply GroupRMSNorm to Q and K if necessary
        if self.apply_qk_norm:
            bq, sq, nhq, hdq = q_states.shape
            bk, sk, nhk, hdk = k_states.shape
            q_states = self.q_norm(q_states.reshape(bq, sq, -1)).reshape(bq, sq, nhq, hdq).to(v_states.device).to(v_states.dtype)
            k_states = self.k_norm(k_states.reshape(bk, sk, -1)).reshape(bk, sk, nhk, hdk).to(v_states.device).to(v_states.dtype)

        # calculate attention mask (sliding window + causal mask)
        sq = q_states.size(1)
        skv = k_states.size(1)
        mask = self.create_attention_mask(sq, skv).to(v_states.device).to(v_states.dtype)

        # when num_kv_heads < num_q_heads, do kv-heads repeating
        # from (batch_size, seq_len, num_kv_heads, head_dim) to (batch_size, seq_len, num_q_heads, head_dim)
        assert(self.num_kv_head <= self.num_q_head)
        num_kv_groups = self.num_q_head // self.num_kv_head
        k_states = torch.repeat_interleave(k_states, dim=2, repeats=num_kv_groups)
        v_states = torch.repeat_interleave(v_states, dim=2, repeats=num_kv_groups)

        # q and k: [b, s, nh, hd] -> [b, nh, s, hd]  (performing dot product per batch per head)
        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)

        # compute the attention logits P = QK^T
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3))  # [b, nh, s_q, s_kv]

        # apply softmax
        attn_weights = self.apply_softmax(attn_weights, mask=mask)

        # apply attention dropout
        if self.softmax_dropout_rate > 0:
            attn_weights = self.apply_dropout(attn_weights)

        # turning v to [b, nh, s_kv, hd]
        v_states = v_states.transpose(1, 2)

        # compute the output by multiplying attention weights with value tensor V
        output = torch.matmul(attn_weights, v_states)  # [b, nh, s_q, hd]

        # reshaping output to [b, s, nh, hd]
        output = output.transpose(1, 2)

        # replacing all-NaN rows with 0s 
        nan_rows = torch.all(torch.isnan(output), dim=-1)
        output[nan_rows] = 0

        return output

    def extract_tensors(self, original_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> list:
        """Extract tensors from a packed tensor, used when QKVLayout is THD"""
        batch_size = cu_seqlens.shape[0] - 1  # batch_size
        original_tensors = [] # len = batch_size
        for i in range(batch_size):
            start, end = cu_seqlens[i], cu_seqlens[i+1]
            original_tensors.append(original_tensor[start:end])
        
        return original_tensors

    def create_attention_mask(self, seq_len_q: int, seq_len_kv: int) -> torch.Tensor:
        """Create sliding window causal attention mask, considering cu_seqlens for variable-length sequences"""
        seq_len = max(seq_len_q, seq_len_kv)
        mask = torch.full((seq_len, seq_len), float('-inf'))
    
        for i in range(seq_len):
            # in the i-th row, only the elements in [i - window_size, i] columns are 0
            if self.window_size is not None and self.causal == True:
                start = max(i - self.window_size, 0)
                mask[i, start:i+1] = 0
            elif self.window_size is None and self.causal == True:
                mask[i, :i+1] = 0
            elif self.window_size is not None and self.causal == False:
                start = max(i - self.window_size, 0)
                end = min(i + self.window_size + 1, seq_len)
                mask[i, start:end] = 0
            else:
                mask[i, :] = 0
        
        # when seq_len_q != seq_len_kv (e.g. in cross-attention)
        # we need to pick the bottom right sub-rectangle of the square mask, returning a mask in [seq_len_q, seq_len_kv]
        if seq_len_q != seq_len_kv:
            if seq_len_q > seq_len_kv:
                mask = mask[:, -seq_len_kv:]
            else:
                mask = mask[-seq_len_q:, :]
        
        return mask
    
    def apply_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply softmax with dot product scaling and possible temperature or capping"""
        # scale the dot product
        logits = logits * self.softmax_scale

        # apply capping
        if self.softmax_cap is not None:
            logits = self.softmax_cap * F.tanh(logits / self.softmax_cap)
        # apply temperature
        else:
            logits = logits / self.softmax_temp

        # apply sliding-window causal mask
        logits = logits + mask

        logits = F.softmax(logits, dim=-1)   # [b, nh, s_q, s_kv], so performing softmax on the head dimension

        # apply clipping
        l, r = self.softmax_clip_range
        logits = torch.clamp((r - l) * logits + l, 0.0, 1.0)
        return logits
    
    def apply_dropout(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply dropout"""
        torch.manual_seed(self.softmax_dropout_seed)
        logits = F.dropout(logits, p=self.softmax_dropout_rate, training=self.training)

        return logits

    
class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.mask = self.create_attention_mask(seqlen_q, seqlen_kv)

        # pad the attention mask
        pad_q = (block_size_q - seqlen_q % block_size_q) % block_size_q
        pad_kv = (block_size_kv - seqlen_kv % block_size_kv) % block_size_kv
        if pad_q > 0 or pad_kv > 0:
            self.mask = F.pad(self.mask, (0, pad_kv, 0, pad_q), value=-float('inf'))
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        # apply GroupRMSNorm to Q and K if necessary
        if self.apply_qk_norm:
            bq, block_q, nhq, hdq = q.shape
            bk, block_k, nhk, hdk = k.shape
            q = self.q_norm(q.reshape(bq, block_q, -1)).reshape(bq, block_q, nhq, hdq).to(v.device).to(v.dtype)
            k = self.k_norm(k.reshape(bk, block_k, -1)).reshape(bk, block_k, nhk, hdk).to(v.device).to(v.dtype)
        
        # calculate the global index span of q, k and v block in the original seq_q and seq_kv
        start_q = block_idx_q * self.block_size_q
        end_q = (block_idx_q + 1) * self.block_size_q
        start_kv = block_idx_kv * self.block_size_kv
        end_kv = (block_idx_kv + 1) * self.block_size_kv

        # extract local mask
        mask = self.mask[start_q : end_q, start_kv : end_kv]
        mask = mask.to(v.device).to(v.dtype)

        # when num_kv_heads < num_q_heads, do kv-heads repeating
        # from (batch_size, block_size, num_kv_heads, head_dim) to (batch_size, block_size, num_q_heads, head_dim)
        assert(self.num_kv_head <= self.num_q_head)
        num_kv_groups = self.num_q_head // self.num_kv_head
        k = torch.repeat_interleave(k, dim=2, repeats=num_kv_groups)
        v = torch.repeat_interleave(v, dim=2, repeats=num_kv_groups)

        # q, k and v: [b, block_size, nh, hd] -> [b, nh, block_size, hd]  (performing dot product per batch per head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute the attention logits P = QK^T
        attn_logits = torch.matmul(q, k.transpose(2, 3))  # [b, nh, block_size_q, block_size_kv]

        # scale the dot product
        attn_logits = attn_logits * self.softmax_scale

        # apply capping
        if self.softmax_cap is not None:
            attn_logits = self.softmax_cap * F.tanh(attn_logits / self.softmax_cap)
        # apply temperature
        else:
            attn_logits = attn_logits / self.softmax_temp

        # apply masking
        attn_logits = attn_logits + mask

        # iterate over block_size_q
        # since we're doing softmax for the seq_len_k dimension (treats it as the X vector in task instructions),
        # the current k-block (lse2) would be the logsumexp of attn_logits[:, :, i, :]
        # and lse1 would be the global_lse[:, :, start_q + i], which stores the logsumexp of all previous k-blocks
        for i in range(self.block_size_q):
            # break if the current q-block exceeds the seqlen_q
            if start_q + i >= self.seqlen_q:
                break

            # extract the local lse for current k-block
            lse2 = torch.logsumexp(attn_logits[:, :, i, :], dim=-1)

            # get global lse for previous k-blocks
            lse1 = global_lse[:, :, start_q + i]

            # used in updating global_o (for update formula, refer to the lecture video on FlashAttn2)
            lse_prev = lse1.clone()

            # update global_lse
            lse_max = torch.max(lse1, lse2)
            lse_min = torch.min(lse1, lse2)
            global_lse[:, :, start_q + i] = lse_max + torch.log1p(torch.exp(lse_min - lse_max))
            
            # used in updating global_o
            lse_cur = global_lse[:, :, start_q + i].clone()

            # update global_o
            # attn_logits[:, :, i, :].shape: [b, nh, block_size_kv]
            # global_lse[:, :, start_q + i].shape: [b, nh] (so need to unsqueeze to [b, nh, 1])
            # attn_output.shape: [b, nh, 1, block_size_kv]
            attn_output = torch.exp(attn_logits[:, :, i, :] - global_lse[:, :, start_q + i].unsqueeze(-1)).unsqueeze(2).to(self.dtype)
            # compute the output block by multiplying attention output with v-block
            output_block = torch.matmul(attn_output, v)  # [b, nh, 1, hd]
            # reshaping output block to [b, 1, nh, hd]
            output_block = output_block.transpose(1, 2)
            # replacing all-NaN rows with 0s
            nan_rows = torch.all(torch.isnan(output_block), dim=-1)
            output_block[nan_rows] = 0
            # update formula: o_i = exp(lse_prev - lse_cur) * o_(i-1) + output_block
            global_o[:, start_q + i, :, :] = (torch.exp(lse_prev - lse_cur)).unsqueeze(-1) * global_o[:, start_q + i, :, :] + output_block.squeeze(1) # [b, nh, hd]