from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, PreTrainedModel, AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model, AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


config = AutoConfig.from_pretrained("gpt2")
config.num_labels = 1
reward_model = AutoModelForSequenceClassification.from_config(config)
print(reward_model)