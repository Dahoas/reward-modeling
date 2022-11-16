import json

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