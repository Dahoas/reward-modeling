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


def clean(texts):
    new_text = []
    for text in texts:
        new_text.append(text.split("Human:")[0])
    return new_text


def infer(model, dataloader, max_length, temp):
    for inputs in tqdm(dataloader):
        tok_prompts = inputs[0]
        data = inputs[1]
        with torch.no_grad():
            outputs = model.generate(**tok_prompts, max_length=max_length, do_sample=True, temperature=temp, sync_gpus=True)[:, tok_prompts["input_ids"].shape[1]:]
        text_outputs = tokenizer.batch_decode(outputs)
        text_outputs = clean(text_outputs)
        for sample, output in zip(data, text_outputs):
            sample["response"] = output
        Logger.log(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset")
    parser.add_argument("--log_file")
    parser.add_argument("--model_name")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--deepspeed", default=False)
    args = parser.parse_args()

    assert args.log_file is not None
    Logger.init(args.log_file)

    # Often taking test split of rm static dataset
    prompt_dataset = load_dataset(args.prompt_dataset)["test"]
    prompt_dataset = [{key: sample[key] for key in sample} for sample in prompt_dataset]
    prompt_dataset = prompt_dataset[:128]

    if args.deepspeed:
        raise NotImplemented
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.eval()
        model.half()
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length = 1024
    def data_collator(data):
        prompts = [sample["prompt"] for sample in data]
        tok_prompts = tokenizer(prompts, padding="longest", return_tensors="pt")
        tok_prompts["input_ids"] = tok_prompts["input_ids"].to("cuda")
        tok_prompts["attention_mask"] = tok_prompts["attention_mask"].to("cuda")
        return tok_prompts, data

    batch_size = 4
    temp = 1.0

    dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)

    infer(model, dataloader, max_length, temp)