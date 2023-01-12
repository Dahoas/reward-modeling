import json
import re
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from copy import deepcopy

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data

def write_jsonl(dataset, filename):
    with open(filename, "w") as f:
        for ele in dataset:
            json.dump(ele, f)
            f.write("\n")

def inspect_output():
    dataset = load_jsonl("augmentations.jsonl")
    sample = dataset[-1]
    with open("inspect.txt", "w") as f:
        f.write(sample["prompt"])
        f.write("\n\n\n\n")
        f.write(sample["response"])


def extract_max_code_block(text : str):
    codeblock_pattern = r'<code>(?s)((?!<code>).)*<\/code>'
    code_block_matches = re.finditer(codeblock_pattern, text)
    code_blocks = []
    for match in code_block_matches:
        start, end = match.span()
        code_block = text[start + 6 : end - 7]  # Want to remove <code>, </code> tags
        code_blocks.append(code_block)
    lengths = torch.tensor([len(block) for block in code_blocks])
    if len(code_blocks) == 0:
        return None
    argmax = torch.argmax(lengths)
    return code_blocks[argmax]

def filter_queried(dataset):
    queried = load_jsonl("full_augmentations.jsonl")
    print(len(queried))
    for query in tqdm(queried):
        flag = False
        QId = query["question_id"]
        Id = query["answer"]["meta_data"]["Id"]
        for i, sample in enumerate(dataset):
            cur_QId = sample["question_id"]
            cur_Id = sample["answer"]["meta_data"]["Id"]
            if QId == cur_QId and Id == cur_Id:
                flag = True
                dataset.pop(i)
                break
        #if not flag:
            #print(QId)
            #print(Id)
            #raise ValueError("Unsupported query")
    return dataset

def filter_instruct_augments():
    dataset = load_jsonl("filtered_full_augmentations.jsonl")
    print("dataset len", len(dataset))
    removal_indices = []
    cnt=0
    for i in tqdm(range(30000, len(dataset))):
        QId = dataset[i]["question_id"]
        Id = dataset[i]["answer"]["meta_data"]["Id"]
        for j in range(i+1, len(dataset)):
            ele_QId = dataset[j]["question_id"]
            ele_Id = dataset[j]["answer"]["meta_data"]["Id"]
            if QId == ele_QId and Id == ele_Id:
                removal_indices.append(j)
                cnt += 1
                print(cnt)
                break
    dataset = [ele for i, ele in enumerate(dataset) if i not in removal_indices]
    #write_jsonl(dataset, "filtered_full_augmentations.jsonl")
    print(cnt)

def upload_dataset():
    dataset = load_jsonl("filtered_full_augmentations.jsonl")
    dict_dataset = {key: [] for key in dataset[0].keys()}
    for ele in dataset:
        for key in dict_dataset:
            dict_dataset[key].append(ele[key])
    hf_dataset = Dataset.from_dict(dict_dataset)
    hf_dataset.push_to_hub("Dahoas/code-review-instruct-critique-revision")

def upload_python_subset():
    dataset = load_dataset("Dahoas/code-review-instruct-critique-revision")
    sample = dataset["train"][0]
    print(sample["meta_data"])
    python_dataset = []
    for ele in tqdm(dataset["train"]):
        if "python" in ele["meta_data"]["Tags"]:
            python_dataset.append(ele)
    print(len(python_dataset))
    python_dict = {key: [] for key in sample.keys()}
    for ele in python_dataset:
        for key in ele.keys():
            python_dict[key].append(ele[key])
    python_dataset = Dataset.from_dict(python_dict)
    python_dataset.push_to_hub("Dahoas/code-review-instruct-critique-revision-python")

if __name__ == "__main__":
    #inspect_output()
    #upload_dataset()
    upload_python_subset()
    #filter_instruct_augments()
