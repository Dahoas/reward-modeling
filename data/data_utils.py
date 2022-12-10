import json

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
