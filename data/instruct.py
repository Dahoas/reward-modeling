import openai
from logger import Logger
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import time

def load_jsonl(filename):
	data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)
			data.append(response)
	return data

def query(prompts, max_tokens):
	responses = openai.Completion.create(engine='text-davinci-003', prompt=prompts, max_tokens=max_tokens, temperature=1.0)["choices"]
	for prompt, response in zip(prompts, responses):
		text = response["text"]
		Logger.log([{"prompt": prompt, "response": text}])
	responses = [response["text"] for response in responses]
	return responses

#questions = load_jsonl("questions.jsonl")
#questions = [question["question"] for question in questions]
#random.shuffle(questions)
# Can use prompts exctracted from hh-rlhf for efficiency

def free_sample():
	prompt = ["Hello", "Goodbye"]
	answer = query(prompt, 1024)
	print(answer)

	'''crit_prompt = "\n Critique the above answer to the question."
	critique = query(prompt + answer + crit_prompt, 250)
	print(critique)
	imp_prompt = "\nIncorporate the above critique to improve the answer."
	improvement = query(prompt + answer + crit_prompt + critique + imp_prompt, 350)
	print(improvement)'''

def gen_prompts():
	hh_prompts = load_jsonl("/home/dahoas/Desktop/datasets/prompts.jsonl")
	hh_prompts = [prompt["prompt"] for prompt in hh_prompts]
	random.shuffle(hh_prompts)
	hh_prompts = list(filter(lambda prompt: len(prompt) < 500, hh_prompts))

	gen_num = 4*1e4 #1e5
	tasks_per_prompt = 10
	prompts_per_query = 10
	num_batches = int(gen_num // ((tasks_per_prompt - 5) * prompts_per_query))

	for _ in tqdm(range(num_batches)):
		inputs = []
		for _ in range(prompts_per_query):
			inds = np.random.choice(len(hh_prompts), 5, replace=False)  # Select 5 random prompts to guide generation
			prompts = [hh_prompts[ind] for ind in inds]
			prompt = "You are a human interacting with a large language model. List {} tasks you want help with. \
			1. {} \
			2. {} \
			3. {} \
			4. {} \
			5. {} \
			".format(tasks_per_prompt, prompts[0], prompts[1], prompts[2], prompts[3], prompts[4])
			inputs.append(prompt)
		try:
			query(inputs, 2048)
		except openai.error.RateLimitError:
			print("RATELIMIT ERROR")
			time.sleep(15)
		time.sleep(15)  # Sleep to prevent rate limiting


def gen_candidates():
	synthetic_prompts = load_jsonl("extracted_synthetic_alignment_prompts.jsonl")
	synthetic_prompts = [prompt["prompt"] for prompt in synthetic_prompts]
	synthetic_prompts = synthetic_prompts[12835 + 24000:]
	#random.shuffle(synthetic_prompts)

	prompts_per_query = 20
	batched_prompts = [synthetic_prompts[i*prompts_per_query : (i+1)*prompts_per_query] for i in range((len(synthetic_prompts) + prompts_per_query - 1) // prompts_per_query)]

	for prompt_batch in tqdm(batched_prompts):
		inputs = []
		for prompt in prompt_batch:
			prompt = "You are a language model. Help complete this task. Task: {} \
".format(prompt)
			# Append prompt twice to get pairwise-comparison down the line
			inputs.append(prompt)
			#inputs.append(prompt)
		try:
			query(inputs, 1024)
		except openai.error.RateLimitError:
			print("RATELIMIT ERROR")
			time.sleep(15)
		time.sleep(10)  # Sleep to prevent rate limiting


def gen_questions():
	return


def gen_preferences():
	hh_pairs = load_jsonl("/home/dahoas/Desktop/datasets/single_context_pairwise.jsonl")  # chosen, rejected
	random.shuffle(hh_pairs)

	gen_num = 1e4 #1e5
	tasks_per_prompt = 1
	prompts_per_query = 20
	num_batches = int(gen_num // ((tasks_per_prompt) * prompts_per_query))
	num_batches = 100

	for _ in tqdm(range(num_batches)):
		inputs = []
		for _ in range(prompts_per_query):
			ind = np.random.choice(len(hh_pairs), 1, replace=False)[0]  # Select 1 random prompts to guide generation
			pair = hh_pairs[ind]
			chosen = pair["chosen"]
			rejected = pair["rejected"]
			dialogue_end = list(filter(lambda x: x >= 0, [-1 if chosen[i] == rejected[i] else i for i in range(min(len(chosen), len(rejected)))]))
			if len(dialogue_end) == 0:
				continue
			else:
				dialogue_end = dialogue_end[0]
			dialogue = chosen[:dialogue_end]
			chosen = chosen[dialogue_end:]
			rejected = rejected[dialogue_end:]
			prompt = "You are a human trying to decide which response best follows from the dialogue. Choose only either Response 1 or Response 2. \n\
\n\
Dialogue: {} \n\
\n\
Reaponse 1: {} \n\
\n\
Response 2: {} \n\
\n\
Choice: ".format(dialogue, chosen, rejected)
			inputs.append(prompt)
		try:
			query(inputs, 2048)
		except openai.error.RateLimitError:
			print("RATELIMIT ERROR")
			time.sleep(15)
		time.sleep(15)  # Sleep to prevent rate limiting

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--log_file")
	args = parser.parse_args()

	Logger.init(args.log_file)
	api_key = ""
        openai.api_key = api_key


	#free_sample()
	Logger.init("davinci_completions")
	gen_candidates()
	#Logger.init("synthetic_alignment_prompts")
	#gen_prompts()
	#gen_preferences()
