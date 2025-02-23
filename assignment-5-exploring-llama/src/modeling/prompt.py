
from typing import Dict, Optional
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from ..utils import find_format_keys


class BatchLayout(Enum):
    """Batch Layout Enum"""
    CONCAT = "concat"
    STACK = "stack"


class PaddingSide(Enum):
    """Padding Side Enum"""
    LEFT = "left"
    RIGHT = "right"


class TruncateSide(Enum):
    """Truncate Side Enum"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PromptType(Enum):
    """Prompt Types Enum"""
    
    SYSTEM = "system"
    CONTEXT = "context"
    QUERY = "query"
    RESPONSE = "response"
    PROMPT = "prompt" # NOTE: prompt = system + context + query
    ALL = "all" # NOTE: all = prompt + response
    

class PromptTemplate(nn.Module):
    """Prompt Template module"""
    
    def __init__(self, template_str: str = ""):
        """Initialize Prompt Template module
        
        Args:
            template_str (str): the template string with the format: "....{key1}...{key2}..."
        """
        super().__init__()
        self.template_str = template_str
        self._keys = set(re.findall(r"\{([^}]+)\}", template_str))
        # before setting the default values, set all keys with placeholder `None`
        self._defaults: Dict[str, Optional[str]] = {key: None for key in self._keys}
    
    def keys(self) -> Dict[str, Optional[str]]:
        """Get the keys with its default values of the prompt template as a dictionary
        NOTE: if any key has not been set with default value, then use `None` as a placeholder
        """
        return self._defaults.copy() # a dictionary with key-value pairs
    
    def set_default(self, **kwargs: Optional[Dict[str, str]]) -> None:
        """Set the default values of the prompt template keys"""
        for key, value in kwargs.items():
            if key in self._keys:
                self._defaults[key] = value
    
    def forward(self, **kwargs: Optional[Dict[str, str]]) -> str:
        """Set the prompt template keys with the given keyword argument to get the formatted prompt
        NOTE:
            1. if certain prompt template key has not been set with its default value, then its corresponding kwarg should be provided
            2. if certain key in the kwargs is not found in the keys of the prompt template, just ignore it
        """
        values = {}
        for key in self._keys:
            if key in kwargs: # keys are passed as kwargs
                values[key] = kwargs[key]
            elif self._defaults[key] is not None: # keys are set with default values
                values[key] = self._defaults[key]
            else:
                raise ValueError(f"Missing value for prompt template key: {key}")

        try:
            return self.template_str.format(**values)
        except KeyError as e:
            raise ValueError(f"Key not found in template or provided kwargs: {e}")