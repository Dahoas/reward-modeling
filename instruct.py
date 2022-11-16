import openai
from logger import Logger
import json
from tqdm import tqdm

Logger.init("prompts_small_log")
api_key = ""

openai.api_key = api_key


def query(prompts, max_tokens):
	response = openai.Completion.create(engine='text-davinci-002', prompt=prompts, max_tokens=max_tokens)
	texts = response["choices"]
	logs = []
	for prompt, text in zip(prompts, texts):
		response = text["text"]
		logs.append({"prompt": prompt, "response": response})
	Logger.log(logs)
	return logs


def gen_supervised(batch_size=1):
	prompts = []
	with open("prompt_small.jsonl","r") as f:
		lines = f.readlines()
		for line in lines:
			prompts.append(json.loads(line)['prompt'])

	prompts = [prompts[batch_size*i : batch_size*(i+1)] for i in range((len(prompts) + batch_size - 1)// batch_size)]
	for prompt in tqdm(prompts):
		response = query(prompt, 150)

if __name__ == "__main__":
	batch_size = 20
	gen_supervised(batch_size)