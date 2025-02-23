from typing import Dict, Any, List, Union, Optional, Tuple, Sequence
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from ...utils import load_jsonl

from .base import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    BaseDatasetConfig,
    BaseDataset,
)


@config_dataclass
class QADatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Question-Answering Tasks"""
    
    question_key: str = make_fixed_field("question")
    answer_key: str = make_fixed_field("answer")
    
    question_prefix: str = make_fixed_field("QUESTION")
    answer_prefix: str = make_fixed_field("ANSWER")
    

class QADataset(BaseDataset):
    """Dataset Class for Question-Answering Tasks"""

    def __init__(
        self,
        config: QADatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize QADataset module
        Args:
            config (QADatasetConfig): qa dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        super().__init__()
        self.config: QADatasetConfig = config
        self.tokenizer = tokenizer
        self.data_files = [data_files] if isinstance(data_files, str) else data_files
        self._samples: List[Dict[str, str]] = []
        for data_file in self.data_files:
            self._samples.extend(load_jsonl(data_file))
        self._num_samples = len(self._samples)
        self._indices: List[int] = list(range(self._num_samples))
        self._num_batches = (self._num_samples + self.config.batch_size - 1) // self.config.batch_size

    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self._num_samples

    def sample(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.
        """
        return self._samples[self._indices[idx]]

    def num_batchs(self) -> int:
        """
        Returns the number of batchs in the dataset.
        """
        return self._num_batches

    def batch(self, idx: int) -> Dict[str, Any]:
        """
        Returns a batch from the dataset.
        """
        start_index = idx * self.config.batch_size
        end_index = min((idx + 1) * self.config.batch_size, self._num_samples)
        batch_indices = self._indices[start_index:end_index]
        batch_samples = [self._samples[i] for i in batch_indices]

        input_texts = []
        labels_list = []  # separate list for labels
        for sample in batch_samples:
            question_text = f"{self.config.prefix_template.forward(prefix=self.config.question_prefix)}{sample[self.config.question_key]}"
            answer_text = f"{self.config.prefix_template.forward(prefix=self.config.answer_prefix)}{sample[self.config.answer_key]}"
            input_text = f"{question_text}{self.config.sep_str}{answer_text}"
            input_texts.append(input_text)
            labels_list.append(answer_text) # labels are the answer text

        tokenized_inputs = self.tokenizer.encode(input_texts)
        tokenized_labels = self.tokenizer.encode(labels_list) # tokenize the answer text for labels

        input_ids = []
        labels = []
        for i in range(len(batch_samples)):
            input_ids.append(tokenized_inputs[i])

            # create labels by padding the answer and prepending ignore_index for the question part
            question_len = len(self.tokenizer.encode(f"{self.config.prefix_template.forward(prefix=self.config.question_prefix)}{batch_samples[i][self.config.question_key]}{self.config.sep_str}"))
            label_ids = [self.config.ignore_idx] * question_len + tokenized_labels[i].tolist()
            labels.append(label_ids)

        padded_input_ids = []
        padded_labels = []
        for ids, label_ids in zip(input_ids, labels):
            # truncate input_ids if necessary
            if len(ids) > self.config.seq_len:
                if self.config.truncate_side == TruncateSide.RIGHT:
                    ids = ids[:self.config.seq_len]
                else:
                    ids = ids[-self.config.seq_len:]

            # pad input_ids
            padding_length_input = self.config.seq_len - len(ids)
            if self.config.padding_side == PaddingSide.RIGHT:
                padded_input_ids.append(ids.tolist() + [self.tokenizer.eos_id] * padding_length_input)
            else:
                padded_input_ids.append([self.tokenizer.bos_id] * padding_length_input + ids.tolist())

            # truncate labels if necessary
            if len(label_ids) > self.config.seq_len:
                if self.config.truncate_side == TruncateSide.RIGHT:
                    label_ids = label_ids[:self.config.seq_len]
                else:
                    label_ids = label_ids[-self.config.seq_len:]

            # pad labels
            padding_length_label = self.config.seq_len - len(label_ids)
            if self.config.padding_side == PaddingSide.RIGHT:
                padded_labels.append(label_ids + [self.config.ignore_idx] * padding_length_label)
            else:
                padded_labels.append([self.config.ignore_idx] * padding_length_label + label_ids)

        input_ids = padded_input_ids
        labels = padded_labels

        labels_tensor = torch.tensor(labels, device=self.config.device)
        input_ids_tensor = torch.tensor(input_ids, device=self.config.device)

        return {
            self.config.input_ids_key: input_ids_tensor,
            self.config.cu_seqlens_key: None,
            self.config.labels_key: labels_tensor,
            self.config.samples_key: batch_samples,
        }

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset, including the samples and batches.

        Args:
            seed (Optional[int], optional): Random seed. Defaults to None to be un-deterministic.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._indices)