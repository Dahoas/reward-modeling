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
    filepath = "next_synthetic_instruct_responses.jsonl"
    dataset = load_jsonl(filepath)
    keys = dataset[0].keys()
    dataset = {key: [sample[key] for sample in dataset] for key in keys}
    dataset = Dataset.from_dict(dataset)
    dataset.push_to_hub("Dahoas/next-synthetic-instruct-responses")

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


########################Formatting prompts to feed into instruct query####################
    

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
    #HERE
    first = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")["train"]
    second = load_dataset("Dahoas/next-synthetic-instruct-responses")["train"]
    prompts = []
    for sample in first:
        human, assistant = sample["prompt"], sample["chosen"]
        prompt = "Human: {} \n Assistant: {} \n\n Above is a dialogue between a human and an assistant. The human is confused about something. Generate a followup comment by the Human and clarifying question by the Human and the Assistant's response. Make sure the Human explains why they asked their question. The assistant can speak any number of sentences. The Human should speak at least two sentences.".format(human, assistant)
        prompts.append(prompt)
    for sample in second:
        human, assistant = sample["prompt"], sample["response"]
        prompt = "Human: {} \n Assistant: {} \n\n Above is a dialogue between a human and an assistant. The human is confused about something. Generate a followup comment by the Human and clarifying question by the Human and the Assistant's response. Make sure the Human explains why they asked their question. The assistant can speak any number of sentences. The Human should speak at least two sentences.".format(human, assistant)
        prompts.append(prompt)
    dataset = Dataset.from_dict({"prompt": prompts})
    dataset.push_to_hub("Dahoas/instruct-human-assistant-prompt")


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

# TODO(dahoas): How to filter garbled responses?
def is_valid(response):
    return True

# Need some way of cleaning self-directives and degeneration
def clean_and_upload_full_synthetic():
    instruct_prompt = "\n\n Above is a dialogue between a human and an assistant. The human is confused about something. Generate a followup comment by the Human and clarifying question by the Human and the Assistant's response. Make sure the Human explains why they asked their question. The assistant can speak any number of sentences. The Human should speak at least two sentences."
    dataset = load_jsonl("first_instruct_human_assistant_gen.jsonl")
    new_dataset = {"prompt": [], "response": []}
    for sample in dataset:
        prompt = sample["prompt"]
        response = sample["response"]
        prompt = prompt.replace(instruct_prompt, "")
        dialog = prompt + response
        split_dialog = dialog.split("Assistant:")
        new_prompt = split_dialog[0]
        split_dialog = split_dialog[1:]
        for response in split_dialog:
            if is_valid(response):
                new_response = "Assistant:" + response.split("Human:")[0]
                new_dataset["prompt"].append(new_prompt)
                new_dataset["response"].append(new_response)
                new_prompt = new_prompt + "Assistant:" + response
            else:
                break

    dataset = Dataset.from_dict(new_dataset)
    dataset.push_to_hub("Dahoas/full-synthetic-hh")

    prompts = new_dataset["prompt"]
    responses = new_dataset["response"]
    sft_size = 40000
    test_size = 10000

    sft_prompts = prompts[:sft_size]
    sft_responses = responses[:sft_size]
    dataset = Dataset.from_dict({"prompt": sft_prompts, "response": sft_responses})
    dataset.push_to_hub("Dahoas/sft-synthetic-hh")
    print(len(dataset))

    rm_prompts = prompts[sft_size:]
    rm_responses = responses[sft_size:]
    train_rm_prompts = rm_prompts[:-test_size]
    test_rm_prompts = rm_prompts[-test_size:]
    train_rm_responses = rm_responses[:-test_size]
    test_rm_responses = rm_responses[-test_size:]
    train_dataset = Dataset.from_dict({"prompt": train_rm_prompts, "response": train_rm_responses})
    test_dataset = Dataset.from_dict({"prompt": test_rm_prompts, "response": test_rm_responses})
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.push_to_hub("Dahoas/rm-synthetic-hh")
    print(len(train_dataset))
    print(len(test_dataset))


def find_responses(prompt, dicts):
    responses = {}
    for key, dataset in dicts.items():
        response = dataset.get(prompt)
        if response is not None:
            dataset.pop(prompt)
        responses[key] = response
    return responses
        

def make_hybrid_rm_dataset_split(split):
    datasets_names = ["Dahoas/pythia_synthetic_125M_inference_{}".format(split), "Dahoas/pythia_synthetic_1B_inference_{}".format(split), "Dahoas/pythia_synthetic_6B_inference_{}".format(split), "Dahoas/pythia_synthetic_20B_inference_{}".format(split)]
    dicts = {}
    for dataset_name in datasets_names:
        dataset = load_dataset(dataset_name)["train"]
        dicts[dataset_name] = {sample["prompt"]: sample["response"] for sample in dataset}


    instruct = load_dataset("Dahoas/rm-synthetic-hh")[split]
    instruct = {sample["prompt"]: sample["response"] for sample in instruct}

    dataset = {"prompt": [], "125M": [], "1B": [], "6B": [], "20B": [], "instruct": []}
    for prompt, instruct_response in tqdm(instruct.items()):
        dataset["prompt"].append(prompt)
        dataset["instruct"].append(instruct_response)
        responses = find_responses(prompt, dicts)
        if None not in responses.values():
            for key, val in responses.items():
                convert = {"Dahoas/pythia_synthetic_125M_inference_train": "125M", "Dahoas/pythia_synthetic_1B_inference_train": "1B", "Dahoas/pythia_synthetic_6B_inference_train": "6B", "Dahoas/pythia_synthetic_20B_inference_train": "20B"}
                key = convert[key]
                dataset[key].append(val)
    print("Len: {}".format(len(dataset["prompt"])))
    dataset = Dataset.from_dict(dataset)
    return dataset

def make_hybrid_rm_dataset():
    dataset = make_hybrid_rm_dataset_split("train")
    dataset.push_to_hub("Dahoas/synthetic_prompt_responses")

convert = {"125M": 1, "1B": 2, "6B": 3, "20B": 4, "instruct": 5}
def m1_less_m2(m1, m2):
    return convert[m1] < convert[m2]

def sample_pair(m1, m2):
    abs(m1 - m2)

def make_pairs(sample):
    pairs = []
    for k1 in sample:
        for k2 in sample:
            if k1 != k2:
                v1 = sample[k1]
                v2 = sample[k2]
                chosen = v2 if m1_less_m2(k1, k2) else v1
                rejected = v1 if m1_less_m2(k1, k2) else v2
                if sample_pair(k1, k2):
                    pairs.append({"chosen": chosen, "rejected": rejected})
    return pairs

# Base dataset 133,00 samples

# Option 1
# 70% of dataset is instruct comparisons
## 40% with neox, 30% with 6B, 0% with 1B, 0% with instruct
# 25% neox comparisons
## 18% with 6B, 7% with 1B
# 5 % 6B comparisons
## 4% with 1B, 1% with 125M
# 133,000 total prompts:
## 133,000 instruct-neox, 
# 334030 total samples
## 133000 instruct-neox, 100209 instruct-6B
## 60125 neox-6B, 23382 neox-1B
## 13361 6B-1B, 3340 6B-125M

# Option 2
## 100,000 instruct-neox, 33,000 instruct-6B disjoint
## neox-6B 50,000 samples (25,000 as with instruct, 25,000 new), neox-1B 25,000 samples (18000 with instruct, 7000 new)
## 6B-1B 25,000 samples (12,500 as with neox, 12,500 new), 5,000 6B-125M (2500 as previous, 2500 new)
## 1B-125M 5,000 samples (2500 as with 6B, 2500 new) 
def make_synthetic_hh_graded_rm():
    dataset = load_dataset("Dahoas/synthetic_prompt_responses")["train"]
    rm_dataset = {"prompt": [], "chosen": [], "rejected": []}

    INSTRUCT_NEOX = 100000
    INSTRUCT_6B = 33000
    NEOX_6B = 50000
    NEOX_6B_SEEN = 25000
    NEOX_6B_NEW = 25000
    assert NEOX_6B == NEOX_6B_NEW + NEOX_6B_SEEN
    NEOX_1B = 25000
    NEOX_1B_SEEN = 18000
    NEOX_1B_NEW = 7000
    assert NEOX_1B == NEOX_1B_NEW + NEOX_1B_SEEN
    SIXB_1B = 25000
    SIXB_1B_NEW = 12500
    SIXB_1B_SEEN = 12500
    assert SIXB_1B == SIXB_1B_NEW + SIXB_1B_SEEN
    SIXB_125M = 5000
    SIXB_125M_NEW = 2500
    SIXB_125M_SEEN = 2500
    assert SIXB_125M == SIXB_125M_NEW + SIXB_125M_SEEN
    ONEB_125M = 5000
    ONEB_125M_NEW = 2500
    ONEB_125M_SEEN = 2500
    assert ONEB_125M == ONEB_125M_NEW + ONEB_125M_SEEN

    # Select instruct-neox, instruct-6B, neox-6B
    dataset = dataset.shuffle()
    new_neox = []
    for i, sample in enumerate(tqdm(dataset)):
        rm_dataset["prompt"].append(sample["prompt"])
        rm_dataset["chosen"].append(sample["instruct"])
        if i < INSTRUCT_NEOX:
            rm_dataset["rejected"].append(sample["20B"])
        else:
            new_neox
            rm_dataset["rejected"].append(sample["6B"])

    # Select neox
    dataset = dataset.shuffle()
    for i, sample in enumerate(tqdm(dataset)):
        if i < NEOX_6B:
            rm_dataset["prompt"].append(sample["prompt"])
            rm_dataset["chosen"].append(sample["instruct"])
            rm_dataset["rejected"].append(sample["20B"])

    dataset = Dataset.from_dict(dataset)
    print("Len: {}".format(len(rm_dataset["prompt"])))
    dataset = dataset.train_test_split(test_size=0.07)
    dataset = DatasetDict({"train": dataset["train"], "test": dataset["test"]})
    dataset.push_to_hub("Dahoas/graded-synthetic-hh-rm")


def upload_augmented_dataset():
    dataset = load_jsonl("text-davinici-001.jsonl")
    print(dataset[0].keys())
    new_dataset = {key: [] for key in dataset[0]}
    new_dataset.pop("response")
    new_dataset.pop("instruct")
    new_dataset["instruct_1"] = []
    new_dataset["instruct_3"] = []
    for sample in dataset:
        for key in sample:
            if key == "response":
                new_key = "instruct_1"
            elif key == "instruct":
                new_key = "instruct_3"
            else:
                new_key = key
            new_dataset[new_key].append(sample[key])
    dataset = Dataset.from_dict(new_dataset)
    print(dataset[0].keys())
    dataset = dataset.train_test_split(test_size=0.05)
    print(len(dataset))
    dataset.push_to_hub("Dahoas/augmented_synthetic_prompt_responses")
        


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
    #gen_human_assistant()
    #clean_and_upload_full_synthetic()
    #make_hybrid_rm_dataset()
    upload_augmented_dataset()
