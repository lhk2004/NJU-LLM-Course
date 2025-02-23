from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum
import os
from glob import glob
from datetime import datetime
from itertools import cycle

from rich import print as rprint
import wandb
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BaseDatasetConfig, BaseDataset

from ..modeling.config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
    make_factory_field,
)

from ..utils import (
    convert_to_list,
    seconds_to_hms_str,
    format_rich_text,
    check_valid_path,
    load_safetensors,
    save_safetensors,
)


class OptimizerType(Enum):
    """Optimizer Types Enum"""
    
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    

class TrainLogType(Enum):
    """Training Log Types Enum"""
    
    TERMINAL = "terminal"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@config_dataclass
class BaseTrainConfig(BaseConfig):
    """Base Training Configurations Dataclass"""
    
    # training configurations
    train_steps: int = make_required_field()
    
    # evaluation configurations
    eval_interval: Optional[int] = None
    eval_steps: int = 0
    
    # transformer configurations
    shuffle: bool = False
    shuffle_seed: Optional[int] = None
    
    # optimizer configurations
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = make_required_field()
    momentum: float = 0. # NOTE: only used for SGD
    betas: Tuple[float, float] = (0.9, 0.999) # NOTE: only used for ADAM & ADAMW
    weight_decay: float = 0. # NOTE: only used for ADAMW
    
    # checkpoint configurations
    load_ckpt_dirs: Optional[Union[str, List[str]]] = None
    load_ckpt_step: bool = True
    
    save_interval: Optional[int] = None
    save_last_step: bool = True
    save_ckpt_dir: str = "." # NOTE: will be created if not exists
    
    max_shard_size: int = 1024 # NOTE: in unit: MB
    step_idx_width: int = 5
    ckpt_step_prefix: str = make_fixed_field("step-")
    ckpt_file_ext: str = make_fixed_field("safetensors")
    
    # logging configurations
    log_interval: Optional[int] = None
    log_last_step: bool = True
    log_types: Tuple[TrainLogType] = (TrainLogType.TERMINAL,)
    log_kwargs: dict = make_factory_field(dict)
    
    # common configurations
    device: str = "cpu"


class BaseTrainer(nn.Module):
    """Base Trainer module
    Define some common APIs for LLM training, \
        and provide the default implementations for these APIs
    """
    
    def __init__(
        self,
        config: BaseTrainConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        """Initialize Base Trainer module
        
        Args:
            config (BaseTrainConfig): Base training configurations
            model (BaseModel): Base model
            tokenizer (BaseTokenizer): Base tokenizer
            train_dataset (BaseDataset): Training dataset
            eval_dataset (Optional[BaseDataset], optional): Evaluation dataset. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # optimizer initialization
        if config.optimizer_type == OptimizerType.ADAMW:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_type == OptimizerType.ADAM:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
            )
        elif config.optimizer_type == OptimizerType.SGD:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

        # load checkpoint if provided
        if config.load_ckpt_dirs:
            self._load_ckpt()

        # logger initialization
        self.loggers = {}
        if TrainLogType.WANDB in config.log_types:
            wandb.init(**config.log_kwargs.get("wandb", {}))
            self.loggers[TrainLogType.WANDB] = wandb
        if TrainLogType.TENSORBOARD in config.log_types:
            tensorboard_log_dir = config.log_kwargs.get(
                "tensorboard", {}
            ).get("log_dir", f"./runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            self.loggers[TrainLogType.TENSORBOARD] = SummaryWriter(
                log_dir=tensorboard_log_dir
            )

        # data loaders
        if config.shuffle:
            self.train_dataset.shuffle(seed=config.shuffle_seed)
            if self.eval_dataset:
                self.eval_dataset.shuffle(seed=config.shuffle_seed)
        self.train_dataloader = cycle(self.train_dataset.batches())
        if self.eval_dataset:
            self.eval_dataloader = cycle(self.eval_dataset.batches())
        self.global_step = 0
    
    def run(self) -> None:
        """Run the whole training steps, \
            until the stopping criterion is met
        NOTE: this is an one-time API, and you have to re-initialize a new trainer if you need to rerun
        """
        rprint(format_rich_text("Start Training...", color="green", bold=True))
        start_time = datetime.now()
        self.model.train()

        for step in range(self.global_step, self.config.train_steps):
            batch = next(self.train_dataloader)
            train_loss = self._train_step(batch)

            # evaluation
            if self.config.eval_interval and (step + 1) % self.config.eval_interval == 0 and self.eval_dataset:
                rprint(format_rich_text(f"Evaluation at step {step + 1}...", color="blue"))
                self.model.eval()
                eval_losses = []
                for _ in range(self.config.eval_steps):
                    eval_batch = next(self.eval_dataloader)
                    eval_loss = self._eval_step(eval_batch)
                    eval_losses.append(eval_loss.item())
                avg_eval_loss = np.mean(eval_losses)
                self._log(step + 1, train_loss=train_loss.item(), eval_loss=avg_eval_loss)
                rprint(format_rich_text(f"Evaluation Loss: {avg_eval_loss:.4f}", color="blue"))
                self.model.train()

            # save checkpoint
            if self.config.save_interval and (step + 1) % self.config.save_interval == 0:
                self._save_ckpt(step + 1)

            # logging
            if self.config.log_interval and (step + 1) % self.config.log_interval == 0:
                self._log(step + 1, train_loss=train_loss.item())

            self.global_step += 1

        # save last checkpoint
        if self.config.save_last_step:
            self._save_ckpt(self.config.train_steps)

        # log last step
        if self.config.log_last_step:
            batch = next(self.train_dataloader)
            train_loss = self._train_step(batch)
            self._log(self.config.train_steps, train_loss=train_loss.item())

        end_time = datetime.now()
        total_time_seconds = (end_time - start_time).total_seconds()
        total_time_hms = seconds_to_hms_str(total_time_seconds)
        rprint(format_rich_text("Start Training...", color="green", bold=True))
        rprint(format_rich_text(f"Total Training Time: {total_time_hms}", color="green"))

        # Clean up loggers
        if TrainLogType.WANDB in self.loggers:
            self.loggers[TrainLogType.WANDB].finish()
        if TrainLogType.TENSORBOARD in self.loggers:
            self.loggers[TrainLogType.TENSORBOARD].close()
    
    def _train_step(
        self,
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        """One training step pass, \
            called in `self.run()`:
            
            1. feed a batch of data to the model to apply forward pass with gradient tracking enabled to get the training loss
            2. apply backward pass to compute the gradients
            3. let the optimizer update the model parameters with the gradients
            
        Args:
            batch_data (Dict[str, Any]): a batch of data as a string-key dictionary
            
        Returns:
            torch.Tensor: the training loss
        """
        self.optimizer.zero_grad()
        batch_data_cuda = {k: v.to(self.config.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**batch_data_cuda)
        loss = outputs
        loss.backward()
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def _eval_step(
        self,
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        """One evaluation step pass, \
            called in `self.run()` when the evaluation criterion is met:
            
            1. feed a batch of data to the model to apply forward pass with gradient tracking disabled to get the evaluation loss
        
        Args:
            batch_data (Dict[str, Any]): a batch of data as a string-key dictionary
            
        Returns:
            torch.Tensor: the evaluation loss
        """
        batch_data_cuda = {k: v.to(self.config.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**batch_data_cuda)
        return outputs
    
    def _load_ckpt(self) -> None:
        """Load the model from the pretrained checkpoint directory (or directories) \
            to resume training if needed, called in `self.__init__()`
        
        NOTE: if multiple checkpoints are provided and the parameter keys are overlapped, \
                the later ones will overwrite the earlier ones
        """
        ckpt_dirs = convert_to_list(self.config.load_ckpt_dirs)
        rprint(format_rich_text(f"Loading checkpoints from {ckpt_dirs}", color="yellow"))
        loaded_step = -1
        for ckpt_dir in ckpt_dirs:
            check_valid_path(ckpt_dir)
            ckpt_pattern = os.path.join(ckpt_dir, f"*.{self.config.ckpt_file_ext}")
            ckpt_files = sorted(glob(ckpt_pattern), key=os.path.getmtime)
            if not ckpt_files:
                rprint(format_rich_text(f"No checkpoint found in {ckpt_dir}", color="red"))
                continue

            latest_ckpt_file = ckpt_files[-1]
            state_dict = load_safetensors(latest_ckpt_file)
            self.model.load_state_dict(state_dict, strict=False)  # Allow partial loading

            if self.config.load_ckpt_step:
                step_file = os.path.join(ckpt_dir, "step.json")
                if os.path.exists(step_file):
                    import json
                    with open(step_file, 'r') as f:
                        step_data = json.load(f)
                        loaded_step = max(loaded_step, step_data.get("step", -1))
                    rprint(format_rich_text(f"Loaded step {loaded_step} from {step_file}", color="yellow"))
                else:
                    rprint(format_rich_text(f"Step index file not found in {ckpt_dir}", color="red"))

        if loaded_step > 0:
            self.global_step = loaded_step
            rprint(format_rich_text(f"Resuming training from step {self.global_step}", color="green"))
    
    def _save_ckpt(self, step: int) -> None:
        """Save the model at the current training step as a checkpoint, \
            called in `self.run()` when the saving criterion is met
        
        Args:
            step (int): current training step
        """
        ckpt_dir = os.path.join(self.config.save_ckpt_dir, f"{self.config.ckpt_step_prefix}{str(step).zfill(self.config.step_idx_width)}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, f"model.{self.config.ckpt_file_ext}")
        save_safetensors(self.model.state_dict(), ckpt_file)
        rprint(format_rich_text(f"Checkpoint saved to {ckpt_file}", color="green"))

        if self.config.load_ckpt_step:
            step_file = os.path.join(ckpt_dir, "step.json")
            import json
            with open(step_file, 'w') as f:
                json.dump({"step": step}, f)
            rprint(format_rich_text(f"Step index saved to {step_file}", color="green"))

    def _log(self, step: int, **kwargs: Any) -> None:
        """Log training information to the specified loggers."""
        if TrainLogType.TERMINAL in self.config.log_types:
            log_str = f"Step: {step}"
            for key, value in kwargs.items():
                log_str += f", {key}: {value:.4f}"
            rprint(format_rich_text(log_str, color="cyan"))

        if TrainLogType.WANDB in self.loggers:
            self.loggers[TrainLogType.WANDB].log({"step": step, **kwargs})

        if TrainLogType.TENSORBOARD in self.loggers:
            for key, value in kwargs.items():
                self.loggers[TrainLogType.TENSORBOARD].add_scalar(key, value, step)
    