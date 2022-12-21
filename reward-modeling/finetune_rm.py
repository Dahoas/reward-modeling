import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
import json
import deepspeed
from rm_datasets import PairwiseDataset, pairwise_data_collator
import argparse
from utils import freeze_bottom_causal_layers, load_yaml, make_rm
from datasets import load_dataset


class SparsePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        PAD_ID = model.PAD_ID
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        chosen = inputs["input_ids"][:bs]
        rejected = inputs["input_ids"][bs:]
        rewards = model(**inputs).logits
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss. Only backprop on last value before padding
        loss = 0
        for i in range(bs):
            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0].item()
            assert divergence_ind > 0
            # Input tokens should be truncated to have endoftext padding
            c_end = (chosen[i] == PAD_ID).nonzero()[0].item()
            r_end = (rejected[i] == PAD_ID).nonzero()[0].item()
            print(c_end, r_end)
            # Index into correct reward
            print(chosen_rewards[i].shape)
            c_truncated_reward = chosen_rewards[i, c_end]
            r_truncated_reward = rejected_rewards[i, r_end]
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        return (loss, outputs) if return_outputs else loss


class DensePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        PAD_ID = model.PAD_ID
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        chosen = inputs["input_ids"][:bs]
        rejected = inputs["input_ids"][bs:]
        rewards = model(**inputs).logits
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss. Only backprop on last value before padding
        loss = 0
        for i in range(bs):
            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0
            # Input tokens should be truncated to have endoftext padding
            c_end = (chosen[i] == PAD_ID).nonzero()[0].item()
            r_end = (rejected[i] == PAD_ID).nonzero()[0].item()
            # Index into correct reward
            c_truncated_reward = chosen_rewards[i][divergence_ind : c_end]
            r_truncated_reward = rejected_rewards[i][divergence_ind : r_end]
            # TODO(dahoas): Probably mean is not the best choice here. Instead want exponential decay
            # as distance from last token increases
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        return (loss, outputs) if return_outputs else loss


class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        PAD_ID = model.PAD_ID
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        chosen = inputs["input_ids"][:bs]
        rejected = inputs["input_ids"][bs:]
        rewards = model(**inputs).logits
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, outputs) if return_outputs else loss


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(**config["train_args"])
    model = make_rm(config["model_path"])
    freeze_bottom_causal_layers(model, config["num_layers_unfrozen"])
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    model.PAD_ID = PAD_ID
    model.config.pad_token_id = model.config.eos_token_id

    data = load_dataset(config["data_path"])
    max_length = 1024
    train_dataset = PairwiseDataset(data["train"], tokenizer, max_length=max_length)
    eval_dataset = PairwiseDataset(data["test"], tokenizer, max_length=max_length)

    training_args = TrainingArguments(**config["train_args"])
    if config["trainer_type"] == "sparse":
        SparsePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=pairwise_data_collator).train()
    elif config["trainer_type"] == "dense":
        DensePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=pairwise_data_collator).train()
    else:
        PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=pairwise_data_collator).train()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path

    train(config)