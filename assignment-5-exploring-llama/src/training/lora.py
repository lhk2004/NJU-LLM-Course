import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BaseDataset

from ..modeling.config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..utils import save_safetensors

from .base import BaseTrainConfig, BaseTrainer


@config_dataclass
class LoRATrainConfig(BaseTrainConfig):
    """LoRA Training Configurations Dataclass"""
    
    lora_weight_A_pattern: str = make_required_field()
    lora_weight_B_pattern: str = make_required_field()
    
    save_only_lora: bool = False
    
    
class LoRATrainer(BaseTrainer):
    """LoRA Trainer module
    Based the common APIs provided by `BaseTrainer`, \
        overwrite some of them to support LoRA fine-tuning
    """
    
    def __init__(
        self,
        config: LoRATrainConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        train_dataset: BaseDataset,
        eval_dataset: BaseDataset,
    ):
        """Initialize LoRA Trainer module
        
        Args:
            config (LoRATrainConfig): LoRA training configurations
            model (BaseModel): Base model
            tokenizer (BaseTokenizer): Base tokenizer
            train_dataset (BaseDataset): Training dataset
            eval_dataset (Optional[BaseDataset], optional): Evaluation dataset. Defaults to None.
        """
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.lora_params = self._find_lora_params()
        self._freeze_base_model()
        self._unfreeze_lora_params()
    
    def _find_lora_params(self):
        lora_params = []
        for name, param in self.model.named_parameters():
            if re.search(self.config.lora_weight_A_pattern, name) or \
               re.search(self.config.lora_weight_B_pattern, name):
                lora_params.append(param)
        return lora_params

    def _freeze_base_model(self):
        for name, param in self.model.named_parameters():
            if not (re.search(self.config.lora_weight_A_pattern, name) or
                    re.search(self.config.lora_weight_B_pattern, name)):
                if param.requires_grad:
                    param.requires_grad = False

    def _unfreeze_lora_params(self):
        for param in self.lora_params:
            param.requires_grad = True
            
    def _save_ckpt(self, step: int) -> None:
        """Save the model as a checkpoint, \
            called in `self.run()` when the saving criterion is met
            
        NOTE: as for LoRA, we can choose to only save the LoRA adapter parameters, \
            since the parameters of the base model are not trainable and can be load from another checkpoint directory individually
        
        Args:
            step (int): current training step
        """
        ckpt_dir = os.path.join(
            self.config.save_ckpt_dir,
            f"{self.config.ckpt_step_prefix}{str(step).zfill(self.config.step_idx_width)}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, f"model.{self.config.ckpt_file_ext}")

        if self.config.save_only_lora:
            lora_state_dict = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.data.cpu()
            save_safetensors(lora_state_dict, ckpt_file, max_shard_size=self.config.max_shard_size)
            print(f"LoRA checkpoint saved to {ckpt_file}")
        else:
            super()._save_ckpt(step)