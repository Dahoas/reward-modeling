import json
import torch
from datasets import Dataset

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


def clean_and_upload(dataset, name):
    cleaned_dataset = {"prompt": [], "response": []}
    for sample in dataset:
        cleaned_prompt = sample["prompt"]
        cleaned_response = sample["response"].split("<|endoftext|>")[0].split("Human:")[0]
        cleaned_dataset["prompt"].append(cleaned_prompt)
        cleaned_dataset["response"].append(cleaned_response)

    dataset = Dataset.from_dict(cleaned_dataset)
    dataset.push_to_hub(name)


if __name__ == "__main__":
    dataset = load_jsonl("pythia_6B_inference_test.jsonl")
    name = "Dahoas/pythia-6B-test-gen"
    clean_and_upload(dataset, name)