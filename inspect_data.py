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

if __name__ == "__main__":
	t2()