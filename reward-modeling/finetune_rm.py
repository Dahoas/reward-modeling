import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
import json
import deepspeed
from rm_datasets import PairwiseDataset, PairwiseEvalDataset, pairwise_data_collator
import argparse
from utils import freeze_bottom_causal_layers, load_yaml, make_rm
from datasets import load_dataset
import wandb


class SparsePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        PAD_ID = model.PAD_ID
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        rewards = model(**inputs)
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, rewards) if return_outputs else loss


class DensePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        PAD_ID = model.PAD_ID
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

    data = load_dataset(config["data_path"])
    max_length = 1024
    train_dataset = PairwiseDataset(data["train"], tokenizer, max_length=max_length, max_num=config["max_train_size"])
    if data.get("test") is not None:
        eval_dataset = PairwiseEvalDataset(data["test"], tokenizer, max_length=max_length)
    else:
        split = data["train"].train_test_split(test_size=0.10)
        train_dataset = PairwiseDataset(split["train"], tokenizer, max_length=max_length)
        eval_dataset = PairwiseEvalDataset(split["test"], tokenizer, max_length=max_length)

    training_args = TrainingArguments(**config["train_args"])
    if config["trainer_type"] == "sparse":
        trainer = SparsePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset, compute_metrics=compute_metrics,
             eval_dataset=eval_dataset, data_collator=pairwise_data_collator)
    elif config["trainer_type"] == "dense":
        trainer = DensePairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            data_collator=pairwise_data_collator)
    else:
        trainer = PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
             data_collator=pairwise_data_collator)
    trainer.train()

    # NOTE: In order to run this install transformers from source
    # per https://github.com/huggingface/transformers/issues/20942
    preds = torch.tensor(trainer.predict(eval_dataset)[0])
    print(preds.shape)
    preds = preds.view(-1, 2)
    samples = {"prompt": [], "chosen": [], "rejected": [], "scores": []}
    for i in range(16):
        ele = data["test"][i]
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
    parser.add_argument("--max_train_size", type=int, default=-1)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path
    config["max_train_size"] = args.max_train_size

    train(config)
