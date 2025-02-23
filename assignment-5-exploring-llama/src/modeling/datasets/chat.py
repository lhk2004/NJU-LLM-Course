from typing import Dict, Any, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from .base import BaseDatasetConfig, PaddingSide, TruncateSide

from .qa import QADataset, QADatasetConfig

@config_dataclass
class ChatDatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Chatbot Tasks"""

    conversations_key: str = make_fixed_field("conversations")
    role_key: str = make_fixed_field("role")
    content_key: str = make_fixed_field("content")

    user_role_value: str = make_fixed_field("user")
    bot_role_value: str = make_fixed_field("chatbot")

    user_role_prefix: str = make_fixed_field("USER")
    bot_role_prefix: str = make_fixed_field("CHATBOT")

class ChatDataset(QADataset):
    """Dataset Class for Chatbot Tasks"""

    def __init__(
        self,
        config: ChatDatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize ChatDataset module
        Args:
            config (ChatDatasetConfig): chat dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        qa_config = QADatasetConfig(
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            batch_layout=config.batch_layout,
            padding_side=config.padding_side,
            truncate_side=config.truncate_side,
            drop_last_incomplete_batch=config.drop_last_incomplete_batch,
            samples_key=config.samples_key,
            input_ids_key=config.input_ids_key,
            labels_key=config.labels_key,
            cu_seqlens_key=config.cu_seqlens_key,
            ignore_idx=config.ignore_idx,
            prefix_template=config.prefix_template,
            sep_str=config.sep_str,
            device=config.device,
        )
        super().__init__(config=qa_config, tokenizer=tokenizer, data_files=data_files)
        self.chat_config: ChatDatasetConfig = config # sse a separate config for ChatDataset

    def sample(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.
        """
        raw_sample = super().sample(idx)
        return raw_sample

    def batch(self, idx: int) -> Dict[str, Any]:
        """
        Returns a batch from the dataset.
        """
        start_index = idx * self.chat_config.batch_size
        end_index = min((idx + 1) * self.chat_config.batch_size, self._num_samples)

        if self.chat_config.drop_last_incomplete_batch and end_index - start_index < self.chat_config.batch_size and end_index < self._num_samples:
            raise IndexError("Batch index out of range due to drop_last_incomplete_batch")

        batch_indices = self._indices[start_index:end_index]
        batch_samples = [self._samples[i] for i in batch_indices]

        input_texts = []
        labels_list = []
        for sample in batch_samples:
            conversations = sample[self.chat_config.conversations_key]
            input_concat = ""
            for turn in conversations:
                role = turn[self.chat_config.role_key]
                content = turn[self.chat_config.content_key]
                if role == self.chat_config.user_role_value:
                    input_concat += f"{self.chat_config.prefix_template.forward(prefix=self.chat_config.user_role_prefix)}{content}{self.chat_config.sep_str}"
                elif role == self.chat_config.bot_role_value:
                    input_concat += f"{self.chat_config.prefix_template.forward(prefix=self.chat_config.bot_role_prefix)}{content}{self.chat_config.sep_str}"
            input_texts.append(input_concat.strip())

            # labels are still the bot's turns
            bot_turns = [
                f"{self.chat_config.prefix_template.forward(prefix=self.chat_config.bot_role_prefix)}{turn[self.chat_config.content_key]}"
                for turn in conversations
                if turn[self.chat_config.role_key] == self.chat_config.bot_role_value
            ]
            labels_list.append(self.chat_config.sep_str.join(bot_turns).strip())

        tokenized_inputs = self.tokenizer.encode(input_texts)
        tokenized_labels = self.tokenizer.encode(labels_list)

        input_ids = []
        labels = []
        for i in range(len(batch_samples)):
            input_ids.append(tokenized_inputs[i])
            labels.append(tokenized_labels[i].tolist())

        padded_input_ids = []
        padded_labels = []
        for ids, label_ids in zip(input_ids, labels):
            # truncate input_ids if necessary
            if len(ids) > self.chat_config.seq_len:
                if self.chat_config.truncate_side == TruncateSide.RIGHT:
                    ids = ids[:self.chat_config.seq_len]
                else:
                    ids = ids[-self.chat_config.seq_len:]

            # pad input_ids
            padding_length_input = self.chat_config.seq_len - len(ids)
            if self.chat_config.padding_side == PaddingSide.RIGHT:
                padded_input_ids.append(ids.tolist() + [self.tokenizer.eos_id] * padding_length_input)
            else:
                padded_input_ids.append([self.tokenizer.bos_id] * padding_length_input + ids.tolist())

            # truncate labels if necessary
            if len(label_ids) > self.chat_config.seq_len:
                if self.chat_config.truncate_side == TruncateSide.RIGHT:
                    label_ids = label_ids[:self.chat_config.seq_len]
                else:
                    label_ids = label_ids[-self.chat_config.seq_len:]

            # pad labels
            padding_length_label = self.chat_config.seq_len - len(label_ids)
            if self.chat_config.padding_side == PaddingSide.RIGHT:
                padded_labels.append(label_ids + [self.chat_config.ignore_idx] * padding_length_label)
            else:
                padded_labels.append([self.chat_config.ignore_idx] * padding_length_label + label_ids)

        input_ids = padded_input_ids
        labels = padded_labels

        labels_tensor = torch.tensor(labels, device=self.chat_config.device)
        input_ids_tensor = torch.tensor(input_ids, device=self.chat_config.device)

        return {
            self.chat_config.input_ids_key: input_ids_tensor,
            self.chat_config.cu_seqlens_key: None,
            self.chat_config.labels_key: labels_tensor,
            self.chat_config.samples_key: batch_samples,
        }