from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation_type = activation_type
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.device = device
        self.dtype = dtype

        # projection matrices
        self.W_up = nn.Parameter(torch.empty(hidden_size, ffh_size, device=device, dtype=dtype))
        self.W_gate = nn.Parameter(torch.empty(hidden_size, ffh_size, device=device, dtype=dtype))
        self.W_down = nn.Parameter(torch.empty(ffh_size, hidden_size, device=device, dtype=dtype))

        # LoRA parameters (if LoRA rank > 0)
        if lora_rank > 0:
            self.A_r = nn.Parameter(torch.empty(hidden_size, lora_rank, device=device, dtype=dtype))
            self.B_r = nn.Parameter(torch.empty(lora_rank, hidden_size, device=device, dtype=dtype))
            self.lora_dropout = nn.Dropout(p=lora_dropout_rate)
        else:
            self.A_r = None
            self.B_r = None

        self.reset_parameters()

    @property
    def activation_fn(self):
        if self.activation_type == MLPActivationType.RELU:
            return F.relu
        elif self.activation_type == MLPActivationType.GELU:
            return F.gelu
        elif self.activation_type == MLPActivationType.SILU:
            return F.silu
        elif self.activation_type == MLPActivationType.SIGMOID:
            return F.sigmoid
        elif self.activation_type == MLPActivationType.BILINEAR:
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        input_dtype = input.dtype
        # MLP(X) = (gate_output * up_output) · W_down, * stands for element wise product and · stands for matrix multiplication
        # gate_output = activation(X · W_gate)
        # up_output = X · W_up
        assert self.dtype == torch.float32, "Param Dtype != float32"
        higher_prec = self.dtype
        gate_output = self.activation_fn(torch.matmul(input.to(higher_prec), self.W_gate.to(higher_prec)))
        up_output = torch.matmul(input.to(higher_prec), self.W_up.to(higher_prec))
        gated_output = gate_output * up_output
        output = torch.matmul(gated_output, self.W_down.to(higher_prec))

        # apply LoRA if applicable
        # MLP_lora(X) = MLP(X) + dropout(X · δW), where δW = (a / r) * (A_r · B_r)
        # intuition: learning task-specific biases
        if self.lora_rank > 0:
            with torch.random.fork_rng():  # not letting lora_dropout_seed interfere with the global random seed
                torch.manual_seed(self.lora_dropout_seed)
                X_times_A_times_B = torch.matmul(input.to(higher_prec), self.A_r.to(higher_prec)).matmul(self.B_r.to(higher_prec))
                lora_output = self.lora_dropout((self.lora_alpha / self.lora_rank) * X_times_A_times_B)
            output += lora_output

        return output.to(input_dtype)
    
    def init_weight(self, weight: torch.Tensor, activation_type: MLPActivationType, is_uniform: bool):
        if activation_type in [MLPActivationType.SIGMOID, MLPActivationType.BILINEAR]:
            if is_uniform:
                nn.init.xavier_uniform_(weight.T)
            else:
                nn.init.xavier_normal_(weight.T)
        else:
            if is_uniform:
                nn.init.kaiming_uniform_(weight.T, nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(weight.T, nonlinearity='relu')
    
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        # initialize the base projection matrices
        torch.manual_seed(self.init_base_seed + 1)
        self.init_weight(self.W_up, self.activation_type, False)
        torch.manual_seed(self.init_base_seed + 2)
        self.init_weight(self.W_gate, self.activation_type, False)
        torch.manual_seed(self.init_base_seed + 3)
        self.init_weight(self.W_down, self.activation_type, False)

        # initialize LoRA weights if applicable
        if self.lora_rank > 0:
            torch.manual_seed(self.lora_init_base_seed + 1)
            self.init_weight(self.A_r, self.activation_type, True)
            torch.manual_seed(self.lora_init_base_seed + 2)
            self.init_weight(self.B_r, self.activation_type, True)

    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation_type = activation_type
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.dtype = dtype
        self.device = device

        self.nle = num_experts // world_size
        assert num_experts % world_size == 0, "num_experts must be divisible by world_size."

        expert_size = ffh_size // num_experts
        assert ffh_size % num_experts == 0, "ffh_size must be divisible by num_experts."

        self.local_experts = nn.ModuleList([
            DenseMLPWithLoRA(
                hidden_size,
                expert_size,
                activation_type,
                init_base_seed=init_base_seed + i,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout_rate=lora_dropout_rate,
                lora_dropout_seed=lora_dropout_seed + i,
                lora_init_base_seed=lora_init_base_seed + i,
                dtype=dtype,
                device=device
            ) for i in range(rank * self.nle, (rank + 1) * self.nle)
        ])

        self.gating = nn.Parameter(torch.empty(hidden_size, num_experts, device=device, dtype=torch.float32))

        self.reset_parameters()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # compute routing probabilities
        logits = input.to(torch.float32) @ self.gating   # [batch_size, seq_len, num_experts]
        routing_probs = F.softmax(logits, dim=-1)

        # get top-k experts
        topk_values, topk_indices = torch.topk(routing_probs, self.moe_topk, dim=-1)  # [batch_size, seq_len, moe_topk]

        # normalize weights
        weights = topk_values / topk_values.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, moe_topk]

        # prepare output tensor
        output = torch.zeros_like(input, device=input.device, dtype=torch.float32)

        # process tokens routed to local experts
        # local_index: [0, 1, ..., nle - 1]
        # global_index: [rank * nle, (rank + 1) * nle - 1]
        for local_idx, expert in enumerate(self.local_experts):
            global_idx = self.rank * self.nle + local_idx
            mask = (topk_indices == global_idx).any(dim=-1)  # mask for tokens routed to this expert, shape [batch_size, seq_len]

            if mask.any():
                # get tokens routed to this expert
                routed_input = input[mask].to(torch.float32)  # [num_routed_tokens, hidden_size]
                expert_output = expert(routed_input)  # [num_routed_tokens, hidden_size]

                # collect weights for these tokens
                routed_weights = weights[mask][topk_indices[mask] == global_idx]

                # apply weights and aggregate outputs
                count = 0
                for i in range(input.shape[0]):
                    for j in range(input.shape[1]):
                        if mask[i, j]:
                            weighted_output = routed_weights[count] * expert_output[count]
                            count += 1
                            output[i, j] += weighted_output

        return output.to(input.dtype)
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        torch.manual_seed(self.init_base_seed)
        nn.init.normal_(self.gating, mean=self.init_mean, std=self.init_std)
