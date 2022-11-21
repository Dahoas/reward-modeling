import os
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

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
training_args = TrainingArguments(output_dir=f'ckpts/{dataset_name}/gpt-neo-four-epoch', num_train_epochs=4, logging_steps=100, save_strategy="epoch",
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
                                  weight_decay=0.01, logging_dir="./logs", fp16=True, bf16=False, learning_rate=5e-6, deepspeed='./ds_config_gpt_2.json', save_total_limit=1)
# gptneo trained in jax

class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
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
            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)
            # Index into correct reward
            c_truncated_reward = chosen_rewards[i][divergence_ind : end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind : end_ind]
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        return (loss, outputs) if return_outputs else loss


model = GPTRewardModel(model_name)
load_checkpoint = False
if load_checkpoint:
    zero_path = ""
    model = load_state_dict_from_zero_checkpoint(model, '')
#model.cuda()




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
            # If chosen is same as rejected just continue
            chosen_encodings_dict = tokenizer('<|startoftext|>' + chosen + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            rejected_encodings_dict = tokenizer('<|startoftext|>' + rejected + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            if len((chosen_encodings_dict["input_ids"] != rejected_encodings_dict["input_ids"]).nonzero()) == 0:
                continue
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]

def data_collator(data):
    return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
            'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}


dataset = PairwiseDataset(data, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator).train()


'''if torch.distributed.get_rank() == 0:
    print("SAVING MODEL")
    dir_path = os.path.join("ckpts", dataset_name)
    if not os.path.isdir(dir_path):
	    os.mkdir(dir_path)
    torch.save(model.state_dict(), os.path.join(dir_path, "model_fp16_8.pt"))'''