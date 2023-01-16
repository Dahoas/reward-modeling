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

def infer(model, dataloader, max_length, temp):
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset")
    parser.add_argument("--log_file")
    parser.add_argument("--model_name")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--split")
    parser.add_argument("--deepspeed", default=False)
    parser.add_argument("--batch_size", default=1)
    args = parser.parse_args()

    assert args.log_file is not None
    Logger.init(args.log_file)

    # Often taking test split of rm static dataset
    prompt_dataset = load_dataset(args.prompt_dataset)[args.split]
    prompt_dataset = [{key: sample[key] for key in sample} for sample in prompt_dataset]
    #prompt_dataset = prompt_dataset[:128]

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()
    model.half()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stop_words_ids = [tokenizer.encode("Human:")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    max_length = 1024
    def data_collator(data):
        prompts = [sample["prompt"] for sample in data]
        tok_prompts = tokenizer(prompts, padding="longest", return_tensors="pt")
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

    infer(model, dataloader, max_length, temp)