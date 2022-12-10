import json
import re

def load_jsonl(filename):
	data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)
			data.append(response)
	return data

def write_jsonl(filename, lines):
	with open(filename, "w") as f:
		for line in lines:
			json.dump(line, f)
			f.write("\n")

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
	write_jsonl("extracted_synthetic_alignment_prompts.jsonl", prompts)


def score_synthetic_preference():
	prefs = load_jsonl("no_prompt_synthetic_hh_preferences.jsonl")
	prefs = [pref["response"] for pref in prefs]
	correct = 0
	for pref in prefs:
		if "Response 1" in pref:
			correct += 1
	print(correct / len(prefs))

if __name__ == "__main__":
	#questions()
	extract_prompts()
	#score_synthetic_preference()