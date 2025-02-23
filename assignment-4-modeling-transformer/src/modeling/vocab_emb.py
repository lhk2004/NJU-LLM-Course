from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class ParallelVocabEmbedding(nn.Module):
    """Parallel Vocab Embedding module
    This is a simplified version of the practical one, \
        which shards the vocabulary embedding table into `world_size` partitions in a process group, and \
        each rank in that process group only handles one partition of it, thus \
        in pratical, we can apply the large vocabulary embedding in parallel and then reduce them together.
    However, for this simpler module, you only need to implement the jobs for any single rank, and \
        don't have to handle the reduce operation or any other parallelism stuff.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        emb_size: int,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Parallel Vocab Embedding module
        
        Args:
            vocab_size(int): vocabulary size
            emb_size(int): embedding size
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean of the normal distribution
            init_std(float, default = 1.0): standard deviation of the normal distribution
            init_base_seed(int, default = 42): the base seed for the initialization (the real seed will be base seed + rank)
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # check if vocab_size is divisible by world_size
        assert vocab_size % world_size == 0, "vocab_size must be divisible by world_size"

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.rank = rank
        self.world_size = world_size
        self.num_ids_per_rank = int(vocab_size / world_size)  # n
        
        # create the embedding table for the given rank
        self.embedding_table = nn.Parameter(torch.empty(self.num_ids_per_rank, emb_size, dtype=dtype, device=device))
        
        # initialize parameters
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        self.reset_parameters()
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """The forward pass for Parallel Vocab Embedding module
        
        Args:
            input_ids(torch.LongTensor): input ids, with shape: (batch_size, seq_len)
        
        Returns:
            output(torch.Tensor): output embedding tensor, with shape: (batch_size, seq_len, emb_size)
        """
        # compute the rank's range of IDs
        lower_bound = self.rank * self.num_ids_per_rank
        upper_bound = (self.rank + 1) * self.num_ids_per_rank
        
        # create an output tensor initialized to zeros
        output = torch.zeros(input_ids.size(0), input_ids.size(1), self.emb_size, dtype=self.embedding_table.dtype, device=input_ids.device)

        # for each input id, retrieve the embedding or keep it zero if out of range
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                id = input_ids[i, j].item()
                if lower_bound <= id and id < upper_bound:
                    output[i, j] = self.embedding_table[id - lower_bound]
        
        return output
        
    def reset_parameters(self) -> None:
        """Initialize learnable embedding parameters for Vocab Embedding from a normal distribution"""
        # set the seed and initialize embedding table
        torch.manual_seed(self.init_base_seed + self.rank)
        nn.init.normal_(self.embedding_table, mean=self.init_mean, std=self.init_std)