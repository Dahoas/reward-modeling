import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
import json
import deepspeed
from rm_datasets import PairwiseDataset, PairwiseEvalDataset, pairwise_data_collator, ranked_data_collator, RankedDataset, RankedEvalDataset
import argparse
from utils import freeze_bottom_causal_layers, load_yaml, make_rm
from datasets import load_dataset
import wandb
import random


class SparsePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        rewards = model(**inputs)
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, rewards) if return_outputs else loss


class RankedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Batch ordered most to least preferable
        rewards = model(**inputs)
        loss = 0
        for i in range(rewards.shape[0]):
            for j in range(i+1, rewards.shape[0]):
                loss += -torch.log(torch.sigmoid(rewards[i] - rewards[j]))
        loss = loss[0] / (rewards.shape[0] * (rewards.shape[0] - 1) / 2)
        return (loss, rewards) if return_outputs else loss


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


def compute_metrics(eval_preds):
    print("EVAL!!!")
    preds = eval_preds.predictions[0].view(-1, 2)
    acc = sum(preds[:, 0] >= preds[:, 1]) / preds.shape[0]
    if torch.distributed.get_rank() == 0:
        wandb.log({"acc": acc})
    return {"accuracy": acc}


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(**config["train_args"])
    model = make_rm(config["model_path"], config["model_type"], config["tokenizer_path"])
    freeze_bottom_causal_layers(model, config["num_layers_unfrozen"])
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    model.PAD_ID = PAD_ID
    model.config.pad_token_id = model.config.eos_token_id
    max_length = 1024

    data = load_dataset(config["data_path"])
    train_data = data["train"]
    if data.get("test") is not None:
        eval_data = data["test"]
    else:
        split = data["train"].train_test_split(test_size=0.05)
        train_data = split["train"]
        eval_data = split["test"]
    
    if config["trainer_type"] == "ranked":
        order = config["order"]
        train_dataset = RankedDataset(train_data, tokenizer, max_length=max_length, order=order, max_num=config["max_train_size"])
        eval_dataset = RankedEvalDataset(eval_data, tokenizer, max_length=max_length, order=order, max_num=config["max_train_size"])
    else:
        train_dataset = PairwiseDataset(train_data, tokenizer, max_length=max_length, max_num=config["max_train_size"])
        eval_dataset = PairwiseEvalDataset(eval_data, tokenizer, max_length=max_length)

    training_args = TrainingArguments(**config["train_args"])
    if config["trainer_type"] == "sparse":
        trainer = SparsePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics,
             eval_dataset=eval_dataset, data_collator=pairwise_data_collator)
    elif config["trainer_type"] == "dense":
        trainer = DensePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            data_collator=pairwise_data_collator)
    elif config["trainer_type"] == "ranked":
        trainer = RankedTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=ranked_data_collator)
    else:
        trainer = PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
             data_collator=pairwise_data_collator)
    trainer.train()

    # NOTE: In order to run this install transformers from source
    # per https://github.com/huggingface/transformers/issues/20942
    if config["trainer_type"] == "ranked":
        num_ranks = len(order)
        preds = torch.tensor(trainer.predict(eval_dataset)[0])
        preds = preds.view(-1, num_ranks)
        samples = {m: [] for m in order}
        samples["prompt"] = []
        samples["scores"] = []
        for i in range(16):
            ele = eval_data[i]
            for m in order:
                samples[m].append(ele[m])
            samples["prompt"].append(ele["prompt"])
            samples["scores"].append(preds[i].tolist())
        # Subtracting rejected scores from chosen scores
        acc = 0
        accs = {}
        convert = {i: m for i, m in enumerate(order)}
        for i in range(num_ranks):
            for j in range(i+1, num_ranks):
                diff = preds[:, i] - preds[:, j]
                local_acc = (diff >= 0).type(torch.float32).sum().item()
                acc += local_acc
                accs[convert[i] + "-" + convert[j]] = local_acc / diff.shape[0]
        acc = acc / (preds.shape[0] * (num_ranks * (num_ranks - 1)) / 2)
        accs["total_acc"] = acc
        print("Testing accuracy: ", acc)
        if torch.distributed.get_rank() == 0:
            wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
            wandb.log(accs)
    else:
        preds = torch.tensor(trainer.predict(eval_dataset)[0])
        preds = preds.view(-1, 2)
        samples = {"prompt": [], "chosen": [], "rejected": [], "scores": []}
        for i in range(16):
            ele = eval_data[i]
            samples["prompt"].append(ele["prompt"])
            samples["chosen"].append(ele["chosen"])
            samples["rejected"].append(ele["rejected"])
            samples["scores"].append(preds[i].tolist())
        # Subtracting rejected scores from chosen scores
        diff = preds[:, 0] - preds[:, 1]
        acc = (diff >= 0).type(torch.float32).mean().item()
        print("Testing accuracy: ", acc)
        if torch.distributed.get_rank() == 0:
            wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
            wandb.log({"acc": acc})
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--max_train_size", type=int, default=-100)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path
    config["max_train_size"] = args.max_train_size

    train(config)
