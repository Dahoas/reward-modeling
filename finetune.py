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

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=3, per_device_eval_batch_size=3, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=False, bf16=True, deepspeed='./ds_config_gpt_j.json')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
model.resize_token_embeddings(len(tokenizer))

data = []
dataset_name = "supervised_chosen_batched"
with open(dataset_name + ".jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        loaded_line = json.loads(line)
        data.append(loaded_line["prompt"])
        #data.append(loaded_line["prompt"] + loaded_line["response"])
print("Len data: ", len(data))

#max_length = 1024
max_length = max([len(tokenizer.encode(text)) for text in data])
print("Max length: {}".format(max_length))

class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = TextDataset(data, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])}).train()
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if torch.distributed.get_rank() == 0:
    print("SAVING MODEL")
    model.save_pretrained("ckpts/" + dataset_name + "/")