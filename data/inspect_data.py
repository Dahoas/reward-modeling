import json

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


if __name__ == "__main__":
	zero_one_label()