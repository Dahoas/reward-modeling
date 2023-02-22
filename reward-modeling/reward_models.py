from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, PreTrainedModel, AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


class RewardModel(nn.Module):
    def __init__(self, config, PAD_ID, save_model):
        super().__init__()
        # TODO(dahoas): fix hacky fix
        model = AutoModelForCausalLM.from_pretrained(config)
        self.config = model.config
        self.neox = "neox" in self.config.model_type
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.gpt_neox if hasattr(model, "gpt_neox") else model.transformer
        dtype = self.config.torch_dtype if hasattr(self.config, "torch_dtype") is not None else torch.float32
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16)
        self.PAD_ID = PAD_ID
        if save_model:
            print("INITIALIZING SELF REFERNCE (FOR ACTIVATION CHECKPOINTING)")
            self.model = model

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss=None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) if self.neox else self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        ends = torch.argmax((input_ids == self.PAD_ID).type(torch.float32), dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)
        return rewards