import json
from tqdm import tqdm
from data_utils import load_jsonl, dump_jsonl, compute_stats
from transformers import AutoTokenizer
import torch
from datasets import load_dataset, Dataset, DatasetDict


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
    data = load_jsonl("synthetic_alignment_prompts.jsonl")
    data = [prompt["response"] for prompt in data]
    prompts = []
    for line in data:
        prompts += re.split("\d\.", line)[1:]
    prompts = [{"prompt": prompt} for prompt in prompts]
    dump_jsonl("extracted_synthetic_alignment_prompts.jsonl", prompts)


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



########################


if __name__ == "__main__":
    #hf_upload()
    #make_hh()
    #make_single_context()
    #make_static()
    #zero_one_label()
    #merge_davinci_gens()
    pair_responses()
    #make_single_context_supervised()
