import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
import json
import argparse
from utils import load_yaml, load_jsonl
from rm_datasets import TextDataset
import wandb


def train(config):

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(**config["train_args"])
    model = AutoModelForCausalLM.from_pretrained(config["model_path"]).cuda()

    data = load_jsonl(config["data_path"])
    print("Len data: ", len(data))

    dataset = TextDataset(data, tokenizer, max_length=max_length)
    train_size = int(0.98 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[2] for f in data])}).train()
    
    
    for batch in val_dataset:
        prompts = tokenizer.batch_decode(batch["input_ids"])
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()
        batch.pop("labels")
        sample_outputs = model.generate(**batch, do_sample=True,
                                        max_length=max_length, top_p=0.95)
        responses = tokenizer.batch_decode(sample_outputs)
        samples += text_output
        for prompt, response in zip(prompts, responses):
            wandb.log({"prompt": prompt, "response": response})
    

    if torch.distributed.get_rank() == 0:
        print("SAVING MODEL")
        model.save_pretrained(config["save_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path

    train(config)