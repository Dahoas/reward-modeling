import json
import torch

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

def compute_stats(dataset):
	#dataset = dataset["train"] if dataset.get("train") is not None else dataset
	stats = {"dataset_len": len(dataset)}
	for column in dataset.features.keys():
		lens_samples = torch.tensor([len(sample) for sample in dataset[column]], dtype=torch.float32)
		avg_len_samples = torch.mean(lens_samples).item()
		std_len_samples = torch.std(lens_samples).item()
		left_tail_samples = torch.sum(lens_samples < avg_len_samples + std_len_samples) / len(lens_samples)
		stats["avg_len_{}".format(column)] = avg_len_samples
		stats["std_len_{}".format(column)] = std_len_samples
		stats["left_tail_{}".format(column)] = left_tail_samples

	print(stats)
	return stats