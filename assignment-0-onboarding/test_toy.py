import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close

from src import matmul_with_importance

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
DEVICE = "cpu"
DTYPE = torch.float32


# configs for each toy test case
toy_test_cases = {
    "case1": {
        "b": 2,
        "s": 5,
        "h": 4,
        "nh": 2,
        "e": 3,
        "top_p": 0.7,
        "top_k": 2,
    }
}


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases.items(),
)
def test_matmul_with_importance(case_key, case_config):
    # define hyper parameters
    b, s, h, nh, e = case_config["b"], case_config["s"], case_config["h"], case_config["nh"], case_config["e"]
    top_p, top_k = case_config["top_p"], case_config["top_k"]
    seed = case_config.pop("seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    device, dtype = case_config.pop("device", DEVICE), case_config.pop("dtype", DTYPE)

    # construct the necessary tensors
    torch.manual_seed(seed)
    input = torch.randn(b, s, h, device=device, dtype=dtype)
    weight = torch.randn(h, e, device=device, dtype=dtype)
    probs = torch.rand(b, s, device=device, dtype=dtype)
    
    # construct the ground-truth output tensor
    if case_key == "case1":
        output_gt = torch.tensor(
            [[[ 1.2396, -0.1764,  1.4309],
            [ 0.6918,  0.8920, -1.4501]],
             
            [[-0.1983,  0.8432, -0.1518],
            [ 0.4134,  0.2719,  0.3612]],
 
            [[ 0.2324,  0.0611,  0.2771],
            [-1.4217, -1.0713, -0.6024]]]
        ).to(device=input.device, dtype=input.dtype)
        grad_input_gt = torch.tensor(
            [[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0490,  0.4652, -0.6257,  0.5394],
          [ 0.0000,  0.0000,  0.0000,  0.0000],
          [-0.6009, -0.1426,  0.2260, -0.5582]],
 
         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [-0.4499,  1.9238, -2.0256,  0.0982],
          [ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]]
        ).to(device=input.device, dtype=input.dtype)
        grad_weight_gt = torch.tensor(
            [[ 0.2021, -1.9632, -0.4662],
            [ 1.2274,  0.1867,  0.5645],
            [ 0.8503,  2.7111,  1.4602],
            [-0.6971, -2.6114, -2.2617]]
        ).to(device=weight.device, dtype=weight.dtype)
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    #----- test if the function works without grad_output -----#
    output, grad_input, grad_weight = matmul_with_importance(
        input=input, 
        weight=weight, 
        probs=probs,
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    
    # check if the output tensor is correct
    assert_close(output, output_gt, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert grad_input is None, "grad_input should be None"
    # check if the grad_weight tensor is correct
    assert grad_weight is None, "grad_weight should be None"
    
    #----- test if the function works with grad_output -----#
    torch.manual_seed(seed)
    grad_output = torch.randn_like(output)
    output, grad_input, grad_weight = matmul_with_importance(
        input=input, 
        weight=weight, 
        probs=probs,
        grad_output=grad_output,
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    
    # check if the output tensor is correct
    assert_close(output, output_gt, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert_close(grad_input, grad_input_gt, atol=atol, rtol=rtol)
    # check if the grad_weight tensor is correct
    assert_close(grad_weight, grad_weight_gt, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main()
    