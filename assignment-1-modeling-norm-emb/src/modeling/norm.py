import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRMSNorm(nn.Module):
    """Group RMS Norm module
    This is a variant of RMS Norm that \
        evenly splits the hidden dimension into groups, and \
        applies root-mean-square normalization with \
            learnable scaling transformation on each i-th group individually.
    """
    
    def __init__(self, 
        hidden_size: int, 
        group_size: int,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Group RMS Norm module
        
        Args:
            hidden_size(int): hidden dimension size
            group_size(int): group size
            eps(float, default = 1e-5): epsilon
            init_range(tuple, default = (-1.0, 1.0)): the range of the uniform distribution to initialize learnable scaling parameters
            init_seed(int, default = 42): seed for the initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()

        # check if hidden_size is divisible by group_size
        if hidden_size % group_size != 0:
            raise ValueError("hidden_size must be divisible by group_size")
        
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = eps
        self.num_groups = int(hidden_size / group_size)
        self.init_seed = init_seed
        self.init_range = init_range
        self.dtype=dtype
        
        # initialize learnable scaling parameter gamma
        self.gamma = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        
        # initialize other parameters
        self.reset_parameters()
        
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """The forward pass for Group RMS Norm module

        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): normalized output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        
        b, s, h = input.shape
        
        # reshape to apply group-wise RMS normalization
        input_grouped = input.view(b, s, self.num_groups, self.group_size)

        # transform to float32
        input_grouped_higher_prec = input_grouped.to(torch.float32)
        
        # compute RMS over group dimension
        rms = torch.rsqrt(input_grouped_higher_prec.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # normalize and scale each group
        normalized = (input_grouped_higher_prec * rms) * self.gamma.view(1, 1, self.num_groups, self.group_size)
        
        # reshape back to original shape
        output = normalized.view(b, s, h)
        output = output.to(input.dtype)
        output = output.to(input.device)

        return output
    
    def reset_parameters(self) -> None:
        """Initialize learnable scaling parameters for Group RMS Norm from a uniform distribution"""
        
        torch.manual_seed(self.init_seed)
        nn.init.uniform_(self.gamma, a=self.init_range[0], b=self.init_range[1])