import json

def load_prompts(file_path):
	data = []
	with open(file_path, "r") as f:
		lines = f.readlines()
		for line in lines:
			loaded_line = json.loads(line)
			data.append(loaded_line["prompt"])
	return data