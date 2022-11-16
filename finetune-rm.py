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
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
import json
from reward_model import GPTRewardModel


class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        rewards = model(**inputs)
        rewards_chunked = rewards.view((2, -1))
        chosen_rewards = rewards_chunked[0]
        rejected_rewards = rewards_chunked[1]
        # compute pairwise loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, outputs) if return_outputs else loss

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.pad_token = tokenizer.eos_token
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, bf16=False, learning_rate=5e-6, deepspeed='./ds_config_gpt_2.json')
# gptneo trained in jax
model = GPTRewardModel("EleutherAI/gpt-neo-2.7B").cuda()

data = []
dataset_name = "single_context_pairwise"
with open(dataset_name + ".jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        loaded_line = json.loads(line)
        data.append(loaded_line)
        #data.append(loaded_line["prompt"] + loaded_line["response"])
print("Len data: ", len(data))

max_length = 1024
#max_length = max([max(len(tokenizer.encode(text["chosen"])), len(tokenizer.encode(text["rejected"]))) for text in data])
print("Max length: {}".format(max_length))


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer('<|startoftext|>' + chosen + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            rejected_encodings_dict = tokenizer('<|startoftext|>' + rejected + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data] + [f[2] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data] + [f[3] for f in data])}


dataset = PairwiseDataset(data, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator).train()


if torch.distributed.get_rank() == 0:
    print("SAVING MODEL")
    dir_path = os.path.join("ckpts", dataset_name)
    if not os.path.isdir(dir_path):
	    os.mkdir(dir_path)
    torch.save(model.state_dict(), os.path.join(dir_path, "model_fp16.pt"))