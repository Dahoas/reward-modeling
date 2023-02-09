from datasets import load_dataset, DatasetDict, Dataset
import os
from tqdm import tqdm
import json
from clean import clean


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


def make_human_comparison_dataset():
    #prompts = load_dataset("Dahoas/hh_eval")["train"]["prompt"]
    prompts = load_dataset("Dahoas/openai_summarize_tldr_human_eval")["train"]["prompt"]
    models = ["125M", "1B", "6B"]

    for model in models:
        d1 = load_dataset("Dahoas/pythia_{}_sft_summarize_eval_human".format(model))["train"]
        d2 = load_dataset("Dahoas/ilql_{}_summarization_eval".format(model))["train"]

        comparison_dataset = {"prompt": [], model: [], "{}_ilql".format(model): []}
        for prompt in tqdm(prompts):
            d1_response = find_reponse_to_prompt(prompt, d1)
            d2_response = find_reponse_to_prompt(prompt, d2)
            if d1_response is None or d2_response is None:
                continue
            d1_response = d1_response["response"]
            d2_response = d2_response["response"]
            comparison_dataset["prompt"].append(prompt)
            comparison_dataset[model].append(d1_response)
            comparison_dataset["{}_ilql".format(model)].append(d2_response)

        dataset = Dataset.from_dict(comparison_dataset)
        print("Len: {}".format(len(dataset)))
        dataset.push_to_hub("Dahoas/{}_summarization_sft_ilql_comparison".format(model))


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


if __name__ == "__main__":
    #make_hh_prompting_baseline_dataset()
    #make_human_prompts()
    #make_human_datasets()
    #make_summarization_eval_prompts()
    make_human_comparison_dataset()
    #make_summary_ilql_responses()
    #make_hh_ilql_responses()
