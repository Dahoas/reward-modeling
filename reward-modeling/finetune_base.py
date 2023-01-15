import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
import json
import argparse
from utils import load_yaml, load_jsonl, freeze_bottom_causal_layers
from rm_datasets import SFTDataset
import wandb
from datasets import load_dataset
import random


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(**config["train_args"])
    model = AutoModelForCausalLM.from_pretrained(config["model_path"]).cuda()

    data = load_dataset(config["data_path"])["train"]
    print("Len data: ", len(data))

    dataset = SFTDataset(data, tokenizer)
    train_size = int(0.98 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[2] for f in data])}).train()

    if torch.distributed.get_rank() == 0:
        if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') != '3':
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
            data = []
            for i in range(16):
                prompt = val_dataset[i][3]
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].view(1, -1).cuda()
                attention_mask = inputs["attention_mask"].view(1, -1).cuda()
                sample_outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, max_length=1024)
                response = tokenizer.batch_decode(sample_outputs)[0].split("<|endoftext|>")[0][len(prompt):]
                data.append([prompt, response])
            cols = ["prompt", "response"]
            wandb.log({"samples": wandb.Table(columns=cols, data=data)})


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