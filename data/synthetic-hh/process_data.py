import json
from tqdm import tqdm
from data_utils import load_jsonl, dump_jsonl, compute_stats
from transformers import AutoTokenizer
import torch
from datasets import load_dataset, Dataset, DatasetDict
import random
import numpy as np
import re


#####hh-rlhf#####

THRESH = 1024
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def read_hh():
    data = []
    static_train = []
    static_test = []
    with open("datasets/hh-rlhf/helpful-base/train.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
            static_train.append(json.loads(line))
    with open("datasets/hh-rlhf/helpful-online/train.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    with open("datasets/hh-rlhf/helpful-rejection-sampled/train.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
            static_train.append(json.loads(line))

    test_data = []
    with open("datasets/hh-rlhf/helpful-base/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            test_data.append(json.loads(line))
            static_test.append(json.loads(line))
    with open("datasets/hh-rlhf/helpful-online/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            test_data.append(json.loads(line))
    with open("datasets/hh-rlhf/helpful-rejection-sampled/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            test_data.append(json.loads(line))
            static_test.append(json.loads(line))
    data += test_data

    return data, static_train, static_test

def extract_prompt_responses(ele):
    chosen = ele["chosen"]
    rejected = ele["rejected"]
    prompt = "Assistant: ".join(chosen.split("Assistant: ")[:-1])
    chosen = "Assistant: " + chosen.split("Assistant: ")[-1]
    rejected = "Assistant: " + rejected.split("Assistant: ")[-1]
    return prompt, chosen, rejected

def extract_single_context_prompt_response(ele):
    chosen = ele["chosen"]
    rejected = ele["rejected"]
    prompt = "Assistant: ".join(chosen.split("Assistant: ")[:-1])
    prompt = "Human: ".join(prompt.split("Human: ")[-2:])
    chosen = "Assistant: " + chosen.split("Assistant: ")[-1]
    rejected = "Assistant: " + rejected.split("Assistant: ")[-1]
    return prompt, chosen, rejected

def make_sft_pariwise(train_dataset, test_dataset, name, sft_size):
    train_dataset = train_dataset.shuffle()
    full_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    full_dataset.push_to_hub("Dahoas/full-{}".format(name))

    # Split full dataset into sft and rm splits
    rm_train_dataset = Dataset.from_dict(train_dataset[sft_size:])
    rm_dataset = DatasetDict({"train": rm_train_dataset, "test": test_dataset})
    rm_dataset.push_to_hub("Dahoas/rm-{}".format(name))

    sft_dataset = Dataset.from_dict(train_dataset[:sft_size]).remove_columns("rejected").rename_column("chosen", "response")
    sft_dataset.push_to_hub("Dahoas/sft-{}".format(name))

    print("{} statistics".format(name))
    compute_stats(train_dataset)
    compute_stats(sft_dataset)
    compute_stats(rm_train_dataset)

# Aggregate all hh-rlhf helpful data and create full dataset, split for sft training, split for reward model training
def make_hh():
    data, _, _ = read_hh()
    formatted_data = {"prompt": [], "chosen": [], "rejected": []}
    for ele in data:
        prompt, chosen, rejected = extract_prompt_responses(ele)
        formatted_data["prompt"].append(prompt)
        formatted_data["chosen"].append(chosen)
        formatted_data["rejected"].append(rejected)

    dataset = Dataset.from_dict(formatted_data)
    split = dataset.train_test_split(test_size=0.10)
    train_dataset = split["train"]
    test_dataset = split["test"]
    make_sft_pariwise(train_dataset, test_dataset, "hh-rlhf", 35000)
    

# In single context we include only the last two rounds of dialogue
def make_single_context():
    data, _, _ = read_hh()
    formatted_data = {"prompt": [], "chosen": [], "rejected": []}
    for ele in data:
        prompt, chosen, rejected = extract_single_context_prompt_response(ele)
        formatted_data["prompt"].append(prompt)
        formatted_data["chosen"].append(chosen)
        formatted_data["rejected"].append(rejected)

    dataset = Dataset.from_dict(formatted_data)
    split = dataset.train_test_split(test_size=0.10)
    train_dataset = split["train"]
    test_dataset = split["test"]
    make_sft_pariwise(train_dataset, test_dataset, "single-context", 35000)


# Dataset without online data tranche
# No need to make single context since most of this data is <1024 tokens
def make_static():
    _, static_train, static_test = read_hh()
    formatted_data = {"prompt": [], "chosen": [], "rejected": []}
    formatted_data_test = {"prompt": [], "chosen": [], "rejected": []}

    for ele in static_train:
        prompt, chosen, rejected = extract_prompt_responses(ele)
        formatted_data["prompt"].append(prompt)
        formatted_data["chosen"].append(chosen)
        formatted_data["rejected"].append(rejected)

    for ele in static_test:
        prompt, chosen, rejected = extract_prompt_responses(ele)
        formatted_data_test["prompt"].append(prompt)
        formatted_data_test["chosen"].append(chosen)
        formatted_data_test["rejected"].append(rejected)

    train_dataset = Dataset.from_dict(formatted_data)
    test_dataset = Dataset.from_dict(formatted_data_test)
    make_sft_pariwise(train_dataset, test_dataset, "static", 20000)


#####Question Generation#####

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
    data = load_jsonl("next_synthetic_prompts.jsonl")
    data = [prompt["response"] for prompt in data]
    prompts = []
    cnt = 0
    for line in data:
        try:
            samples = re.split("\d\.", line)[2:]
            for sample in samples:
                prompts.append(sample.strip())
        except:
            cnt += 1
    print(cnt)
    dataset = Dataset.from_dict({"prompt": prompts})
    dataset.push_to_hub("Dahoas/next_synthetic_prompts")


def score_synthetic_preference():
    prefs = load_jsonl("no_prompt_synthetic_hh_preferences.jsonl")
    prefs = [pref["response"] for pref in prefs]
    correct = 0
    for pref in prefs:
        if "Response 1" in pref:
            correct += 1
    print(correct / len(prefs))


#####rm-data####

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

def hf_upload():
    filepath = "datasets/synthetic/gptj_completions.jsonl"
    dataset = load_jsonl(filepath)
    dataset = {"prompt": [prompt["prompt"] for prompt in dataset], "response": [response["response"] for response in dataset]}
    dataset = Dataset.from_dict(dataset)
    dataset.push_to_hub("Dahoas/sft-gptj-synthetic-prompt-responses")

def pair_responses():
    dataset_one = "Dahoas/instruct-synthetic-prompt-responses"
    dataset_two = "Dahoas/sft-gptj-synthetic-prompt-responses"
    instruct_data = load_dataset(dataset_one)
    gptj_data = load_dataset(dataset_two)

    print(instruct_data)
    print(gptj_data)

    instruct_data = instruct_data["train"]
    gptj_data = gptj_data["train"]
    instruct_data = {instruct_data[i]["prompt"]: instruct_data[i]["response"] for i in range(len(instruct_data))}
    gptj_data = {gptj_data[i]["prompt"]: gptj_data[i]["response"] for i in range(len(gptj_data))}

    prompts = []
    chosen = []
    rejected = []
    for prompt, response in tqdm(instruct_data.items()):
        for gptj_prompt, gptj_response in gptj_data.items():
            if gptj_prompt in prompt:
                prompts.append(prompt)
                chosen.append(response)
                rejected.append(gptj_response)
                gptj_data.pop(gptj_prompt)
                break

    print(len(prompts))
    
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})
    dataset.push_to_hub("Dahoas/synthetic-instruct-gptj-pairwise")

def adjust_pairwise():
    def clean(text):
        text = text.strip()
        if len(text) > 0:
            if not text[0].isalpha():
                text = text[1:]
            if len(text) > 0 and not text[-1].isalpha():
                text = text[:-1]
        return text.strip() + "."

    dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")["train"]
    prompts, chosen, rejected = [], [], []
    for sample in dataset:
        prompts.append(clean(sample["prompt"]))
        chosen.append(clean(sample["chosen"]))
        rejected.append(clean(sample["rejected"]))
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})
    dataset.push_to_hub("Dahoas/synthetic-instruct-gptj-pairwise")


########################Formatting prompts to feed into instruct query#####################


def gen_prompt_format_dataset():
    hh_prompts = load_jsonl("../datasets/prompts.jsonl")
    hh_prompts = [prompt["prompt"] for prompt in hh_prompts]
    synthetic_prompts = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")["train"]["prompt"]
    prompts = hh_prompts + synthetic_prompts
    random.shuffle(prompts)
    prompts = list(filter(lambda prompt: len(prompt) < 500, prompts))

    gen_num = 8*1e4 #1e5
    tasks_per_prompt = 10
    prompts_per_query = 10
    num_batches = int(gen_num // ((tasks_per_prompt - 5) * prompts_per_query))

    inputs = []
    for _ in tqdm(range(num_batches)):
        for _ in range(prompts_per_query):
            inds = np.random.choice(len(prompts), 5, replace=False)  # Select 5 random prompts to guide generation
            examples = [prompts[ind] for ind in inds]
            prompt = "You are a human interacting with a large language model. List {} tasks you want help with. \
            1. {} \
            2. {} \
            3. {} \
            4. {} \
            5. {} \
            ".format(tasks_per_prompt, examples[0], examples[1], examples[2], examples[3], examples[4])
            inputs.append(prompt)
    print(len(inputs))
    dataset = Dataset.from_dict({"prompt": inputs})
    dataset.push_to_hub("Dahoas/hh_prompt_format")


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

def gen_human_assistant():
    dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")["train"]
    prompts = []
    for sample in dataset:
        human, assistant = sample["prompt"], sample["chosen"]
        prompt = "Human: {} \n Assistant: {} \n\n Above is a dialogue between a human and an assistant. The human is confused about something. Generate a followup comment by the Human and clarifying question by the Human and the Assistant's response. Make sure the Human explains why they asked their question. The assistant can speak any number of sentences. The Human should speak at least two sentences.".format(human, assistant)
        prompts.append(prompt)
    dataset = Dataset.from_dict({"prompt": prompts})
    dataset.push_to_hub("Dahoas/first-instruct-human-assistant-prompt")


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


if __name__ == "__main__":
    #hf_upload()
    #make_hh()
    #make_single_context()
    #make_static()
    #zero_one_label()
    #merge_davinci_gens()
    #pair_responses()
    #adjust_pairwise()
    #make_single_context_supervised()
    #gen_prompt_format_dataset()
    #gen_responses()
    #extract_prompts()
    gen_human_assistant()
