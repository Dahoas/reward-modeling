import json
from tqdm import tqdm
from data_utils import load_jsonl, dump_jsonl

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
	data = []
	dataset_name = "single_context_pairwise"
	with open(dataset_name + ".jsonl", "r") as f:
		lines = f.readlines()
		for line in lines:
			loaded_line = json.loads(line)
			data.append(loaded_line)
			#data.append(loaded_line["prompt"] + loaded_line["response"])
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
	#zero_one_label()
        #merge_davinci_gens()
        pair_instruct_gptj()
