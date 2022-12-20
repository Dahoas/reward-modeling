import os
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '9994'
#os.environ['RANK'] = "0"
#os.environ['LOCAL_RANK'] = "0"
#os.environ['WORLD_SIZE'] = "4"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel
import json
from reward_model import GPTRewardModel
import deepspeed

data = []
dataset_name = "single_context_pairwise"
with open(dataset_name + ".jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        loaded_line = json.loads(line)
        data.append(loaded_line)
        #data.append(loaded_line["prompt"] + loaded_line["response"])
print("Len data: ", len(data))



tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
training_args = TrainingArguments(output_dir=f'ckpts/{dataset_name}/gpt-j', num_train_epochs=8, logging_steps=100, save_strategy="epoch",
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, bf16=False, learning_rate=5e-6, deepspeed='./ds_config_gpt_j.json', save_total_limit=1)
# gptneo trained in jaxh

model = GPTRewardModel("EleutherAI/gpt-j-6B")
layers = model.transformer.h
num_layers = len(layers)
num_unfrozen = int(0.5 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)
load_checkpoint = False
if load_checkpoint:
    model.load_state_dict(torch.load('ckpts/single_context_pairwise/model_fp16.pt'))
#model.cuda()


max_length = 1024
#max_length = max([max(len(tokenizer.encode(text["chosen"])), len(tokenizer.encode(text["rejected"]))) for text in data])
print("Max length: {}".format(max_length))







dataset = PairwiseDataset(data, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator).train()