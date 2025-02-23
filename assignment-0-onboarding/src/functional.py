from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """

    if not input.requires_grad:
        input = input.clone().detach().requires_grad_(True)
    if not weight.requires_grad:
        weight = weight.clone().detach().requires_grad_(True)
        
    # basic shapes definitions
    b, s, h = input.shape
    e = weight.shape[1]
    hd = h // num_heads  # Dimension per head

    # applying importance filtering using top_p and top_k

    # mask based on top_p (keep elements with probs >= top_p)
    mask = (probs >= top_p)

    # mask base on top_k (keep only top_k highest probabilities)
    if top_k is not None:
        top_k_vals, top_k_indices = torch.topk(probs, k=top_k, dim=1, sorted=False)
        top_k_mask = torch.zeros_like(mask, dtype=torch.bool)
        top_k_mask.scatter_(1, top_k_indices, 1)
        mask = mask & top_k_mask  # (b, s)

    # gathering important elements based on the mask
    important_indices = mask.nonzero(as_tuple=True)
    important_input = input[important_indices]   # (total_important_seq_len, h)

    # performing matmul on important elements only
    t = important_input.size(0)  # total_important seq_len
    important_input = important_input.view(t, num_heads, hd)
    W2 = weight.view(num_heads, hd, e)
    output = torch.einsum('bnh,nhe->bne', important_input, W2)  # (t, nh, e)

    # computing gradients if grad_output is provided
    grad_input = None
    grad_weight = None
    if grad_output is not None:
        grad_input, grad_weight = torch.autograd.grad(
            outputs=output, 
            inputs=[input, weight], 
            grad_outputs=grad_output, 
            retain_graph=True
        )
    return output, grad_input, grad_weight
        
        
    
    