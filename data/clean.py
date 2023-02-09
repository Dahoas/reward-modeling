import json
import torch
from datasets import Dataset
import os

def load_prompts(file_path):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line["prompt"])
    return data

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data

def dump_jsonl(filename, data):
    with open(filename, "w") as f:
        for dict_t in data:
                json.dump(dict_t, f)
                f.write("\n")

def clean(text):
    clean_text = text.split("<|endoftext|>")[0].split("Human:")[0]
    split = clean_text.split("Assistant:")
    if len(split) > 2:
        print("Split is too long!")
        split = split[:2]
    return "Assistant:" + split[-1]


def clean_and_upload(dataset, name):
    cleaned_dataset = {"prompt": [], "response": []}
    for sample in dataset:
        # Add space since there is space at start of hh prompt
        cleaned_prompt = sample["prompt"]
        cleaned_response = "Assistant:" + sample["response"].split("<|endoftext|>")[0].split("Human:")[0].split("Assistant:")[-1]
        cleaned_dataset["prompt"].append(cleaned_prompt)
        cleaned_dataset["response"].append(cleaned_response)

    dataset = Dataset.from_dict(cleaned_dataset)
    dataset.push_to_hub(name)


def clean_rm_and_upload(dataset, name):
    cleaned_dataset = {"response": [], "reward": []}
    for sample in dataset:
        response = sample["prompt"].replace("<|endoftext|>", "")
        cleaned_dataset["response"].append(response)
        cleaned_dataset["reward"].append(sample["reward"])
    dataset = Dataset.from_dict(cleaned_dataset)
    dataset.push_to_hub(name)


if __name__ == "__main__":
    '''files = os.listdir("datasets")
    for file in files:
        dataset = load_jsonl(os.path.join("datasets", file))
        name = file.replace(".jsonl", "")
        print("Processing {}...".format(name))
        clean_and_upload(dataset, name)'''
    dataset = load_jsonl("6B_rm_inference_train.jsonl")
    clean_rm_and_upload(dataset, "Dahoas/reward-labeled-static")