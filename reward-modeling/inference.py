from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import deepspeed
from datasets import load_dataset
from logger import Logger
from accelerate import Accelerator
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from utils import make_rm
import random


def load_rm(model_name, tokenizer_name, model_path, save_model):
    rm = make_rm(model_name, "causal", tokenizer_name, save_model)
    rm.load_state_dict(torch.load(model_path), strict=True)
    return rm


def clean(texts):
    new_text = []
    for text in texts:
        new_text.append(text.split("Human:")[0])
    return new_text

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

stopping_criteria = None

def infer_clm(model, dataloader, max_length, temp):
    
    for inputs in tqdm(dataloader):
        tok_prompts = inputs[0]
        data = inputs[1]
        with torch.no_grad():
            if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') != '3':
                # Note: synced_gpus=True only needed for zero3
                outputs = model.generate(**tok_prompts, max_length=max_length, do_sample=True, temperature=temp, stopping_criteria=stopping_criteria)[:, tok_prompts["input_ids"].shape[1]:]
            else:
                outputs = model.generate(**tok_prompts, max_length=max_length, do_sample=True, temperature=temp, stopping_criteria=stopping_criteria, synced_gpus=True)[:, tok_prompts["input_ids"].shape[1]:]
        text_outputs = tokenizer.batch_decode(outputs)
        text_outputs = clean(text_outputs)
        for sample, output in zip(data, text_outputs):
            sample["response"] = output
        Logger.log(data)


def infer_rm(model, dataloader, max_length, temp):
    for inputs in tqdm(dataloader):
        tok_prompts = inputs[0]
        data = inputs[1]
        with torch.no_grad():
            outputs = model(**tok_prompts)
            outputs = outputs.tolist()
        for sample, output in zip(data, outputs):
            sample["reward"] = output[0]
        Logger.log(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset")
    parser.add_argument("--log_file")
    parser.add_argument("--model_name")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--split")
    parser.add_argument("--deepspeed", default=False)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--rm_path", default="")
    parser.add_argument("--order", nargs="*", default=["response"])
    parser.add_argument("--save_model", default=False, action="store_true")
    args = parser.parse_args()

    #print("Order: {}".format(args.order))

    assert args.log_file is not None
    assert args.save_model is not None
    Logger.init(args.log_file)

    # Often taking test split of rm static dataset
    dataset = load_dataset(args.prompt_dataset)[args.split]
    dataset = [{key: sample[key] for key in sample} for sample in dataset]
    #prompt_dataset = prompt_dataset[:128]

    if args.rm_path != "":
        model = load_rm(args.model_name, args.tokenizer_name, args.rm_path, args.save_model)
        prompt_dataset = []
        for m in args.order:
            prompt_dataset += [{"prompt": sample["prompt"] + sample[m] + "<|endoftext|>", "type": m, "id": i} for i, sample in enumerate(dataset)]
        #random.shuffle(prompt_dataset)
    else:
        prompt_dataset = dataset
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()
    model.half()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    if args.rm_path != "":
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    stop_words_ids = [tokenizer.encode("Human:")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    max_length = 1024 + 128
    def data_collator(data):
        prompts = [sample["prompt"] for sample in data]
        tok_prompts = tokenizer(prompts, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        tok_prompts["input_ids"] = tok_prompts["input_ids"].to("cuda")
        tok_prompts["attention_mask"] = tok_prompts["attention_mask"].to("cuda")
        return tok_prompts, data

    batch_size = int(args.batch_size)
    temp = 1.0

    dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)
    model = model.to(accelerator.device)

    if args.rm_path != "":
        infer_rm(model, dataloader, max_length, temp)
    else:
        infer_clm(model, dataloader, max_length, temp)