import json
from tqdm import tqdm
from data_utils import load_jsonl, dump_jsonl, compute_stats
from transformers import AutoTokenizer
import torch
from datasets import load_dataset, Dataset


#####hh-rlhf#####

THRESH = 1024
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def read_hh():
	data = []
	with open("datasets/hh-rlhf/helpful-base/train.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			data.append(json.loads(line))
	with open("datasets/hh-rlhf/helpful-online/train.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			data.append(json.loads(line))
	with open("datasets/hh-rlhf/helpful-rejection-sampled/train.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			data.append(json.loads(line))

	test_data = []
	with open("datasets/hh-rlhf/helpful-base/test.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			test_data.append(json.loads(line))
	with open("datasets/hh-rlhf/helpful-online/test.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			test_data.append(json.loads(line))
	with open("datasets/hh-rlhf/helpful-rejection-sampled/test.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			test_data.append(json.loads(line))
	data += test_data

	return data

# Aggregate all hh-rlhf helpful data and create full dataset, split for sft training, split for reward model training
def make_hh():
	data = read_hh()
	formatted_data = {"prompt": [], "chosen": [], "rejected": []}
	for ele in data:
		chosen = ele["chosen"]
		rejected = ele["rejected"]
		prompt = "".join(chosen.split("Assistant: ")[:-1]).replace("Human: ", "")
		chosen = chosen.split("Assistant: ")[-1]
		rejected = rejected.split("Assistant: ")[-1]
		formatted_data["prompt"].append(prompt)
		formatted_data["chosen"].append(chosen)
		formatted_data["rejected"].append(rejected)

	dataset = Dataset.from_dict(formatted_data)
	splits = dataset.train_test_split(test_size=0.05)
	print(splits)
	splits.push_to_hub("Dahoas/full-hh-rlhf")

	# Split full dataset into sft and rm splits
	sft_size = 40000
	dataset = dataset.shuffle()
	rm_data = Dataset.from_dict(dataset[sft_size:])
	splits = rm_data.train_test_split(test_size=0.05)
	print(splits)
	splits.push_to_hub("Dahoas/rm-hh-rlhf")

	sft_data = Dataset.from_dict(dataset[:sft_size]).remove_columns("rejected").rename_column("chosen", "response")
	print(sft_data)
	sft_data.push_to_hub("Dahoas/sft-hh-rlhf")


# Make prompt dataset: Just take first dialogue from human
def make_hh_prompts(data):
	prompts = []
	for line in data:
		prompt = line['chosen'].split('Human:')[1].split('Assistant:')[0]
		prompts.append({'prompt': prompt})

	with open("prompts.jsonl","w") as f:
		for prompt in prompts:
			json.dump(prompt, f)
			f.write('\n')

	print(len(prompts))
	

def make_hh_supervised(data):
	prompts = []
	for line in tqdm(data):
		chosen = line["chosen"]
		# Extract the last two dialogue blocks
		chosen_dialogue = "".join(chosen.split("Human:")[1:][-2:]).replace("Assistant:", "")
		if len(tokenizer(chosen_dialogue)["input_ids"]) > 1024:
			# Only take last dialogue block if last two is too long
			chosen_dialogue = "".join(chosen.split("Human:")[1:][-1]).replace("Assistant:", "")
		prompts.append({"prompt": chosen_dialogue})

	name = "single_context_chosen"
	with open(name+".jsonl","w") as f:
		for prompt in prompts:
			json.dump(prompt, f)
			f.write('\n')

	print(len(prompts))
	lens = torch.tensor([len(prompt['prompt']) for prompt in prompts], dtype=torch.float32)
	avg_len = torch.mean(lens).item()
	std_len = torch.std(lens).item()
	left_tail = torch.sum(lens < avg_len + std_len) / len(lens)
	stats = {
		"avg_len": avg_len,
		"std_len": std_len,
		"left_tail": left_tail
	}
	print(stats)


# In single context we include only the last two rounds of dialogue
def make_single_context():
	data = read_hh()
	formatted_data = {"prompt": [], "chosen": [], "rejected": []}
	for ele in data:
		chosen = ele["chosen"]
		rejected = ele["rejected"]
		prompt = "".join(chosen.split("Assistant: ")[:-1])
		prompt = "".join(prompt.split("Human: ")[-2:])
		chosen = chosen.split("Assistant: ")[-1]
		rejected = rejected.split("Assistant: ")[-1]
		formatted_data["prompt"].append(prompt)
		formatted_data["chosen"].append(chosen)
		formatted_data["rejected"].append(rejected)

	dataset = Dataset.from_dict(formatted_data)
	splits = dataset.train_test_split(test_size=0.05)
	print(splits)
	compute_stats(dataset)
	splits.push_to_hub("Dahoas/full-single-context")

	# Split full dataset into sft and rm splits
	sft_size = 40000
	dataset = dataset.shuffle()
	rm_data = Dataset.from_dict(dataset[sft_size:])
	splits = rm_data.train_test_split(test_size=0.05)
	print(splits)
	compute_stats(rm_data)
	splits.push_to_hub("Dahoas/rm-single-context")

	sft_data = Dataset.from_dict(dataset[:sft_size]).remove_columns("rejected").rename_column("chosen", "response")
	print(sft_data)
	compute_stats(sft_data)
	sft_data.push_to_hub("Dahoas/sft-single-context")


def test_double_counting():
	sample_prompt = data[0]["chosen"].split("Human: ")[1].split("Assistant: ")[0]
	i = 0
	print(i)
	print(sample_prompt)
	for line in data:
		if sample_prompt in line["chosen"]:
			i+=1
			print(i)
			print(line["chosen"])


#####Questions#####

def questions():
	data = []
	with open("new_prompts.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)["response"]
			data.append(response)

	with open("questions.jsonl", "w") as f:
		for response in data:
			questions = re.split("\d.", response)[1:]
			for question in questions:
				json.dump({"question": question}, f)
				f.write("\n")

def extract_prompts():
	data = load_jsonl("synthetic_alignment_prompts.jsonl")
	data = [prompt["response"] for prompt in data]
	prompts = []
	for line in data:
		prompts += re.split("\d\.", line)[1:]
	prompts = [{"prompt": prompt} for prompt in prompts]
	dump_jsonl("extracted_synthetic_alignment_prompts.jsonl", prompts)


def score_synthetic_preference():
	prefs = load_jsonl("no_prompt_synthetic_hh_preferences.jsonl")
	prefs = [pref["response"] for pref in prefs]
	correct = 0
	for pref in prefs:
		if "Response 1" in pref:
			correct += 1
	print(correct / len(prefs))


#####rm-data#####

def find_break_ind(chosen, rejected):
	for i in range(min(len(chosen), len(rejected))):
		if chosen[i] != rejected[i]:
			return i
	return None

def make_single_context_supervised():
	data = load_jsonl("datasets/single_context/single_context_pairwise.jsonl")
	dump = []
	for ele in tqdm(data):
		chosen, rejected = ele["chosen"], ele["rejected"]
		break_ind = find_break_ind(chosen, rejected)
		if break_ind is None:
			continue
		prompt = chosen[:break_ind]
		response = chosen[break_ind:]
		dump.append({"prompt": prompt, "response": response})
	dump_jsonl("datasets/single_context/single_context_supervised.jsonl", dump)


def t1():
	data = []
	with open("prompts.jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			data.append(json.loads(line))
		print(len(lines))

	data_tiny = data[:10]
	data_small = data[:1000]
	data_medium = data[:10000]
	with open("prompts_tiny.jsonl","w") as f:
		print(len(data_tiny))
		for line in data_tiny:
			json.dump(line, f)
			f.write('\n')
	with open("prompt_small.jsonl","w") as f:
		print(len(data_small))
		for line in data_small:
			json.dump(line, f)
			f.write('\n')

def t2():
	dataset_name = "single_context_pairwise.jsonl"
	data = load_jsonl(dataset_name)
	print("Len data: ", len(data))
	cnt = 0
	repeated = []
	for line in data:
		chosen = line["chosen"]
		rejected = line["rejected"]
		if chosen == rejected:
			cnt += 1
			repeated.append(chosen)
	print(cnt)
	print(repeated[0])

def zero_one_label():
	data = []
	dataset_name = "single_context_pairwise"
	with open(dataset_name + ".jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			loaded_line = json.loads(line)
			data.append(loaded_line)

	for data_element in data:
		data_element["chosen_reward"] = 1
		data_element["rejected_reward"] = 0

	with open("single_context_pairwise_binary_reward.jsonl", "w") as f:
		for line in data:
			json.dump(line, f)
			f.write("\n")


#####Synthetic#####

def merge_davinci_gens():
    data_1 = load_jsonl("datasets/synthetic_completions.jsonl")
    data_2 = load_jsonl("datasets/davinci_completions.jsonl")

    data = data_1 + data_2

    # Filtering for redundant prompts
    data_dict = {}
    for datapoint in data:
        prompt = datapoint["prompt"].split("Task: ")[1]
        data_dict[prompt] = datapoint["response"]
    data = [{"prompt": k, "response": v} for k, v in data_dict.items()]
    dump_jsonl("complete_davinci_completions.jsonl", data)

def pair_instruct_gptj():
    instruct_data = load_jsonl("datasets/complete_davinci_completions.jsonl")
    instruct_data = {datapoint["prompt"]: datapoint["response"] for datapoint in instruct_data}
    gptj_data = load_jsonl("datasets/gptj_completions.jsonl")
    gptj_data = {datapoint["prompt"]: datapoint["response"] for datapoint in gptj_data}

    data = []
    for prompt, response in tqdm(instruct_data.items()):
        for gptj_prompt, gptj_response in gptj_data.items():
            if prompt in gptj_prompt:
                data.append({"chosen": response, "rejected": gptj_response})
                gptj_data.pop(gptj_prompt)
                break
    dump_jsonl("datasets/instruct_gptj_pairs.jsonl", data)

if __name__ == "__main__":
	make_hh()
	#make_single_context()
	#zero_one_label()
	#merge_davinci_gens()
	#pair_instruct_gptj()
	#make_single_context_supervised()
