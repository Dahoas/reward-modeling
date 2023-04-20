from datasets import load_dataset, DatasetDict, Dataset
import os
from tqdm import tqdm
import json
from clean import clean
import numpy as np
import random
from transformers import AutoTokenizer
from clean import dump_jsonl


def load_jsonl(filename):
	data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)
			data.append(response)
	return data


def make_human_prompts():
    hh_test = load_dataset("Dahoas/rm-static")["test"].shuffle(seed=42)
    synthetic_test = load_dataset("Dahoas/rm-synthetic-hh")["test"].shuffle(seed=42)
    # Pick human eval subsets
    human_eval_size = 100

    hh_human_eval = {"prompt": hh_test[:human_eval_size]["prompt"]}
    hh_eval = Dataset.from_dict(hh_human_eval)
    hh_eval.push_to_hub("Dahoas/hh_eval")

    def fix_ends(texts):
        new_texts = []
        for text in texts:
            prefix = text[:-5]
            tail = text[-5:].replace("1", "")
            new_prompt = prefix + tail
            new_prompt = new_prompt.replace("  ", "")
            new_texts.append(new_prompt)
        return new_texts
    synthetic_human_eval = {"prompt": fix_ends(synthetic_test[:human_eval_size]["prompt"])}
    synthetic_eval = Dataset.from_dict(synthetic_human_eval)
    synthetic_eval.push_to_hub("Dahoas/synthetic_eval")


    hh_synthetic = {"prompt": hh_human_eval["prompt"][:50] + synthetic_human_eval["prompt"][:50]}
    hh_synthetic_eval = Dataset.from_dict(hh_synthetic)
    synthetic_eval.push_to_hub("Dahoas/hh_synthetic_eval")

def find_reponse_to_prompt(prompt, dataset):
    for sample in dataset:
        # Still need to do prompted evals
        #temp_sample = sample["prompt"][:-4].replace("Q:", "Human:").replace("A:", "Assistant:")
        #snippet = temp_sample.split("Human:")[-1]

        if prompt == sample["prompt"]:
        #if snippet in prompt:
            #sample["response"] = sample["response"].replace("Q:", "Human:").replace("A:", "Assistant:")
            return sample
    print("No response found for prompt: \n\n {}".format(prompt))
    return None


def make_human_datasets():
    hh_eval = load_dataset("Dahoas/hh_eval")["train"]["prompt"]
    synthetic_eval = load_dataset("Dahoas/synthetic_eval")["train"]["prompt"]
    hh_synthetic_eval = load_dataset("Dahoas/hh_synthetic_eval")["train"]["prompt"]
    summarization_eval = load_dataset("Dahoas/openai_summarize_tldr_human_eval")["train"]["prompt"]

    path = "inference/sft_summarization_eval"
    datasets = os.listdir(path)
    for f in datasets:
        print(f)
        file_path = os.path.join(path, f)
        dataset = load_jsonl(file_path)
        new_dataset = {"prompt": [], "response": []}
        if "hybrid" in file_path:
            search_dataset = hh_synthetic_eval
        elif "synthetic" in file_path:
            search_dataset = synthetic_eval
        elif "summar" in file_path:
            search_dataset = summarization_eval
        else:
            search_dataset = hh_eval
        for prompt in tqdm(search_dataset):
            sample = find_reponse_to_prompt(prompt, dataset)
            if sample is None:
                continue
            new_dataset["prompt"].append(prompt)
            cleaned_response = clean(sample["response"])
            new_dataset["response"].append(cleaned_response)
        new_dataset = Dataset.from_dict(new_dataset)
        dataset_name = f[:-6] + "_human"
        print("Uploading {}...".format(dataset_name))
        print(len(new_dataset))
        print(new_dataset[0])
        new_dataset.push_to_hub("Dahoas/{}".format(dataset_name))



def make_hh_prompting_baseline_dataset():
    dataset = load_dataset("Dahoas/hh_eval")["train"]["prompt"]
    new_dataset = {"prompt": []}
    for prompt in dataset:
        prompt = prompt.replace("Human:", "Q:").replace("Assistant:", "A:") + "\n\nA:"
        new_dataset["prompt"].append(prompt)
    new_dataset = Dataset.from_dict(new_dataset)
    new_dataset.push_to_hub("Dahoas/hh_prompted_baseline_prompts")


def make_summarization_eval_prompts():
    dataset = load_dataset("CarperAI/openai_summarize_tldr")["test"]
    eval_dataset = {"prompt": dataset["prompt"][:100]}
    eval_dataset = Dataset.from_dict(eval_dataset)
    eval_dataset.push_to_hub("Dahoas/openai_summarize_tldr_human_eval")

def make_hh_ilql_responses():
    dataset = load_dataset("reciprocate/hh_eval_ilql")["train"]
    keys = [key for key in dataset[0].keys() if key != "prompt"]
    print(keys)
    for key in keys:
        new_dataset = {"prompt": [], "response": []}
        for sample in dataset:
            new_dataset["prompt"].append("Assistant:".join(sample["prompt"].split("Assistant:")[:-1]))
            new_dataset["response"].append(sample[key])
        new_dataset = Dataset.from_dict(new_dataset)
        name = key.split("_")[-1]
        print(name)
        new_dataset.push_to_hub("Dahoas/ilql_{}_hh_eval".format(name))

def make_summary_ilql_responses():
    dataset = load_dataset("reciprocate/summarize_eval_ilql")["train"]
    keys = [key for key in dataset[0].keys() if key != "prompt"]
    print(keys)
    for key in keys:
        new_dataset = {"prompt": [], "response": []}
        for sample in dataset:
            new_dataset["prompt"].append(sample["prompt"])
            new_dataset["response"].append(sample[key])
        new_dataset = Dataset.from_dict(new_dataset)
        name = key.split("_")[-1]
        print(name)
        new_dataset.push_to_hub("Dahoas/ilql_{}_summarization_eval".format(name))


def clean_augmented_synthetic_prompt_responses():
    dataset = load_dataset("Dahoas/augmented_synthetic_prompt_responses", revision="b4f4b90660d8ba6a7c3ce59e8af659923834dbb1")
    def f(dataset):
        new_dataset = {key: [] for key in dataset[0].keys()}
        for sample in tqdm(dataset):
            for key in sample:
                if key == "prompt":
                    new_dataset[key].append(sample[key].strip() + "\n\nAssistant:")
                else:
                    new_dataset[key].append(" " + sample[key].replace("Assistant:", "").strip())
        return Dataset.from_dict(new_dataset)
    train_dataset = f(dataset["train"])
    test_dataset = f(dataset["test"])
    new_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    new_dataset.push_to_hub("Dahoas/augmented_synthetic_prompt_responses") 

def make_instruct_preference_queries():
    dataset = load_dataset("Dahoas/augmented_synthetic_prompt_responses")

    def f(dataset):
        new_dataset = {"prompt": [], "dialogue": [], "instruct_1": [], "instruct_3": []}
        for sample in tqdm(dataset):
            dialogue, instruct_1, instruct_3 = sample["prompt"], sample["instruct_1"], sample["instruct_3"]
            new_dataset["dialogue"].append(dialogue)
            new_dataset["instruct_1"].append(instruct_1)
            new_dataset["instruct_3"].append(instruct_3)
            responses = [instruct_1, instruct_3]
            random.shuffle(responses)
            response_a, response_b = responses
            prompt = "Below is a dialogue between a user asking a question and an assistant. At the end you will find two possible responses from the Assistant. Select either Reponse A or Response B as the most helpful response to the user's dialogue.\n\n\n\
Dialogue: {}\n\n\n\
Response A:{}\n\n\n\
Response B:{}\n\n\n\
Choose either Response A or Response B as being the most helpful:".format(dialogue, response_a, response_b)
            new_dataset["prompt"].append(prompt)
        return Dataset.from_dict(new_dataset)

    train_dataset = f(dataset["train"])
    test_dataset = f(dataset["test"])
    new_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    print(new_dataset["train"][0])
    new_dataset.push_to_hub("Dahoas/instruct_preference_queries")


def analyze_instruct_preference():
    dataset = load_jsonl("synthetic-hh/instruct_preference_queries_test.jsonl")
    cnt_1, cnt_3 = 0, 0
    for sample in tqdm(dataset):
        prompt = sample["prompt"]
        response_a = prompt.split("Response A:")[1].split("Response B:")[0]
        response_b = prompt.split("Response B:")[1].split("\n\n\n")[0]
        if "Response A" in sample["response"] and "Response B" not in sample["response"]:
            chosen = response_a
        elif "Response A" not in sample["response"] and "Response B" in sample["response"]:
            chosen = response_b
        else:
            print("Conflicting response: {}".format(sample["response"]))
        if sample["instruct_3"] in chosen:
            cnt_3 += 1
        elif sample["instruct_1"] in chosen:
            cnt_1 += 1
    print(cnt_3 / len(dataset))
    print(cnt_1 / len(dataset))


def shp_processing_pipeline():
    dataset = load_dataset("stanfordnlp/SHP")
    tok = AutoTokenizer.from_pretrained("gpt2")

    # Need to filter out responses below 2.0 score
    print("Filtering by score...")
    dataset = dataset.filter(lambda sample: sample["score_ratio"] > 2.0)

    # Truncate histories greater than 1024 chars or throw out entirely
    MAX_LEN = 1024 - (len(tok("Human: ").input_ids) + len(tok("\n\nAssistant:").input_ids))
    print("Filtering by context length...")
    def compute_r_len(sample):
        sample["r_len"] = max(len(tok(sample["human_ref_A"]).input_ids), len(tok(sample["human_ref_B"]).input_ids))
        return sample
    dataset = dataset.map(compute_r_len, num_proc=24)
    dataset = dataset.filter(lambda sample: sample["r_len"] < MAX_LEN)
    def truncate(sample):
        max_history_len = MAX_LEN - sample["r_len"]
        sample["history"] = "Human: " + tok.decode(tok(sample["history"]).input_ids[-max_history_len:]) + "\n\nAssistant:"
        return sample
    print("Truncating histories...")
    dataset = dataset.map(truncate, num_proc=24)
    
    # Need to only select 5 responses from each dialogue
    histories = {}
    def make_pair_id(sample):
        return sample["post_id"] + sample["c_root_id_A"] + sample["c_root_id_B"]
    for sample in tqdm(dataset["train"]):
        history = histories.get(sample["post_id"])
        if history is None:
            histories[sample["post_id"]] = [make_pair_id(sample)]
        elif len(history) < 5:
            histories[sample["post_id"]].append(make_pair_id(sample))
    train_dataset = dataset["train"].filter(lambda sample: make_pair_id(sample) in histories[sample["post_id"]])
    print("Filtered size: ", len(train_dataset))

    dataset = DatasetDict({"train": train_dataset, "test": dataset["test"]})

    # Format for RM repo
    def form(sample):
        if sample["score_A"] < sample["score_B"]:
            temp = sample["score_A"]
            sample["score_A"] = sample["score_B"]
            sample["score_B"] = temp
        return sample
    dataset = dataset.map(form)
    dataset = dataset.rename_column("history", "prompt")
    dataset = dataset.rename_column("human_ref_A", "chosen")
    dataset = dataset.rename_column("human_ref_B", "rejected")
    print(dataset["train"][0])

    dataset.push_to_hub("Dahoas/filtered-SHP")


def make_instruct_preferences():
    train = load_jsonl("synthetic-hh/instruct_preference_queries_train.jsonl")
    test = load_jsonl("synthetic-hh/instruct_preference_queries_test.jsonl")

    def f(dataset):
        new_dataset = {"prompt": [], "response": [], "chosen": [], "rejected": []}
        fails = 0
        for sample in dataset:
            prompt = sample["prompt"]
            response_a = prompt.split("Response A:")[1].split("Response B:")[0]
            response_b = prompt.split("Response B:")[1].split("\n\n\n")[0]
            if "Response A" in sample["response"] and "Response B" not in sample["response"]:
                chosen = response_a
                rejected = response_b
            elif "Response A" not in sample["response"] and "Response B" in sample["response"]:
                chosen = response_b
                rejected = response_a
            else:
                fails += 1
                continue
                print("Unable to parse response: {}".format(sample["response"]))
                

            if sample["instruct_3"] in chosen and sample["instruct_1"] in rejected:
                chosen = sample["instruct_3"]
                rejected = sample["instruct_1"]
            elif sample["instruct_1"] in chosen and sample["instruct_3"] in rejected:
                chosen = sample["instruct_1"]
                rejected = sample["instruct_3"]
            else:
                fails += 1
                continue
                print("###############Input response not found###############")
                print("Response A:")
                print(response_a)
                print("Response B:")
                print(response_b)
                print("Instruct_3:")
                print(sample["instruct_3"])
                print("Instruct_1:")
                print(sample["instruct_1"])
                print("######################################################")

            new_dataset["prompt"].append(sample["dialogue"])
            new_dataset["chosen"].append(chosen)
            new_dataset["rejected"].append(rejected)
            new_dataset["response"].append(chosen)
                
        print("Extraction fail rate: ", fails / len(dataset))
        print("Old len: ", len(dataset))
        print("New len: ", len(new_dataset["prompt"]))
        dataset = Dataset.from_dict(new_dataset)
        return dataset

    train = f(train)
    test = f(test)
    print(train[0])
    print(test[0])
    dataset = DatasetDict({"train": train, "test": test})
    dataset.push_to_hub("Dahoas/instruct_helpful_preferences")


def make_hh_human_eval():
    dataset = load_dataset("Dahoas/static-hh")["test"]
    dataset = dataset.select([i for i in range(100)])
    dataset = dataset.remove_columns(["response", "chosen", "rejected"])
    dataset = DatasetDict({"train": dataset})
    print(len(dataset))
    print(dataset["train"][1])
    print(dataset)
    dataset.push_to_hub("Dahoas/hh_human_eval")

    dataset = load_dataset("Dahoas/hh_human_eval")


def make_hh_prompted_eval():
    dataset = load_dataset("Dahoas/hh_human_eval")
    print(dataset["train"][1])
    def f(sample):
        sample["prompt"] = sample["prompt"].replace("Human:", "Q:").replace("Assistant:", "A:")
        return sample
    dataset = dataset.map(f)
    print(dataset["train"][1])
    dataset.push_to_hub("Dahoas/hh_prompted_human_eval")


def fix_datasets():
    def replace(sample):
        sample["prompt"] = sample["prompt"].replace("\n\nAssistant ", "\n\nAssistant: ")
        return sample
    datasets = ["Dahoas/hh_human_eval", "Dahoas/static-hh", "Dahoas/rm-static"]
    for dataset_name in datasets:
        print(dataset_name)
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(replace)
        print(dataset["train"][0])
        dataset.push_to_hub(dataset_name)


def hf_dataset_to_jsonl(dataset):
    jsonl = []
    for i in range(len(dataset["prompt"])):
        jsonl.append({key: val[i] for key, val in dataset.items()})
    return jsonl


def make_human_comparison_dataset():
    prompts = load_dataset("Dahoas/hh_human_eval")["train"]["prompt"]
    #prompts = load_dataset("Dahoas/openai_summarize_tldr_human_eval")["train"]["prompt"]
    models = ["125M", "1B", "6B", "20B"]
    r_type = "ppo"

    for model in models:
        d1 = load_jsonl("logs/pythia_sft_{}_hh_eval.jsonl".format(model))
        d2 = load_jsonl("logs/pythia_{}_{}_hh_eval.jsonl".format(r_type, model))

        comparison_dataset = {"prompt": [], model: [], "{}_{}".format(r_type, model): []}
        for prompt in tqdm(prompts):
            d1_response = find_reponse_to_prompt(prompt, d1)
            d2_response = find_reponse_to_prompt(prompt, d2)
            if d1_response is None or d2_response is None:
                continue
            d1_response = d1_response["response"]
            d2_response = d2_response["response"]
            comparison_dataset["prompt"].append(prompt)
            comparison_dataset[model].append(d1_response)
            comparison_dataset["{}_{}".format(r_type, model)].append(d2_response)

        #dataset = Dataset.from_dict(comparison_dataset)
        dataset = hf_dataset_to_jsonl(comparison_dataset)
        print("Len: {}".format(len(dataset)))
        print(dataset[0])
        dump_jsonl("comparisons/hh_human_eval_{}_{}.jsonl".format(r_type, model), dataset)
        #dataset.push_to_hub("Dahoas/{}_summarization_sft_ilql_comparison".format(model))


def make_rl_prompt_dataset():
    dataset1 = load_dataset("Dahoas/instruct_helpful_preferences")
    dataset2 = load_dataset("Dahoas/static-hh")

    from datasets import concatenate_datasets

    dataset = concatenate_datasets([dataset1["train"], dataset2["train"]]).shuffle()
    dataset = DatasetDict({"train": dataset, "test": dataset2["test"]})
    dataset.push_to_hub("Dahoas/rl-prompt-dataset")


if __name__ == "__main__":
    #make_hh_prompting_baseline_dataset()
    #make_human_prompts()
    #make_human_datasets()
    #make_summarization_eval_prompts()
    #make_human_comparison_dataset()
    #make_summary_ilql_responses()
    #make_hh_ilql_responses()
    #clean_augmented_synthetic_prompt_responses()
    #make_instruct_preference_queries()
    #analyze_instruct_preference()
    #shp_processing_pipeline()
   # make_instruct_preferences()
   #make_hh_human_eval()
   #make_hh_prompted_eval()
   #fix_datasets()
   #make_human_comparison_dataset()
   #make_human_comparison_dataset()
   make_rl_prompt_dataset()