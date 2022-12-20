import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel
import json
from reward_model import RewardModel
import deepspeed
from rm_datasets import PairwiseDataset
import argparse
from utils import freeze_bottom_causal_layers


def make_rm(model_name):
    config = AutoConfig.from_pretrained("gpt2")
    config.num_labels = 1
    reward_model = AutoModelForSequenceClassification.from_config(config)
    return reward_model


class SparsePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        model.PAD_ID = PAD_ID
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        chosen = inputs["input_ids"][:bs]
        rejected = inputs["input_ids"][bs:]
        rewards = model(**inputs)
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
            c_truncated_reward = chosen_rewards[i][c_end]
            r_truncated_reward = rejected_rewards[i][r_end]
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        return (loss, outputs) if return_outputs else loss


class DensePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, PAD_ID, return_outputs=False):
        # forward pass
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


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(**config["train_args"])

    data = load_dataset(config["data_path"])
    max_length = 1024
    train_dataset = PairwiseDataset(data["train"], tokenizer, max_length=max_length)
    eval_dataset = PairwiseDataset(data["eval"], tokenizer, max_length=max_length)

    model = RewardModel(model_name)
    freeze_bottom_causal_layers(model, config["num_layers_unfrozen"])

    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    model.PAD_ID = PAD_ID
    training_args = TrainingArguments(**config["train_args"])
    #TrainingArguments(output_dir=f'ckpts/{dataset_name}/gpt-neo-four-epoch', num_train_epochs=4, logging_steps=100, save_strategy="epoch",
    #                                per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
    #                                weight_decay=0.01, logging_dir="./logs", fp16=True, bf16=False, learning_rate=5e-6, deepspeed='./ds_config_gpt_2.json', save_total_limit=1)
    if config["use_sparse_trainer"]:
        SparsePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=data_collator).train()
    else:
        DensePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=data_collator).train()

    

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