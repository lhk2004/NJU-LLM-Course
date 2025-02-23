import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len   # ms
        self.base = base
        self.ratio = ratio
        self.dynamic = dynamic
        self.dtype = dtype
        self.device = device
        
        # precompute initial (C, S)
        self.compute_positional_embeddings(max_seq_len)
        
        # register as non-persistent buffers (not as learnable params)
        self.register_buffer('cos', self.C, persistent=False)
        self.register_buffer('sin', self.S, persistent=False)
    
    def compute_positional_embeddings(self, seq_len: int):
        """Compute cosine and sine positional embeddings."""
        self.max_seq_len = seq_len
        
        base = self.base * self.ratio ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=self.dtype, device=self.device) / self.dim))

        t = torch.arange(self.max_seq_len, device=self.device, dtype=self.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.C = torch.cos(emb)
        self.S = torch.sin(emb)

        self.register_buffer('cos', self.C, persistent=False)
        self.register_buffer('sin', self.S, persistent=False)
        
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int, optional): The offset of the starting position index of the input tensor. Defaults to 0.
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = input.size()

        # check if seq_len exceeds the maximum extended length
        if seq_len > self.max_seq_len:
            # calculate new ratio
            k_ = 0
            for new_k in range(self.ratio, seq_len):
                if new_k * self.max_seq_len >= seq_len and new_k % 2 == 0:
                    k_ = new_k
                    break
            assert k_ != 0, "Cannot find a new scaling ratio."
            new_max_seq_len = self.max_seq_len * k_

            # update internal state if dynamic
            if self.dynamic:
                self.ratio = k_
                self.compute_positional_embeddings(new_max_seq_len)
            else:
                # compute temporary C_ and S_
                base = self.base * k_ ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=self.dtype, device=self.device) / self.dim))

                t = torch.arange(new_max_seq_len, device=self.device, dtype=self.dtype)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                C_ = torch.cos(emb)
                S_ = torch.sin(emb)

                return apply_rotary_pos_emb(input, C_[offset : seq_len + offset], S_[offset : seq_len + offset])
        
        # apply rotary positional embedding 
        return apply_rotary_pos_emb(input, self.cos[offset : seq_len + offset], self.sin[offset : seq_len + offset])
