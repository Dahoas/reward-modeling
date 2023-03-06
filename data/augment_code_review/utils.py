import json
import re
import torch
from datasets import Dataset

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data

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
    for query in queried:
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
        assert flag
    return dataset

def upload_dataset():
    dataset = load_jsonl("../full_augmentations.jsonl")
    dict_dataset = {key: [] for key in dataset[0].keys()}
    for ele in dataset:
        for key in dict_dataset:
            dict_dataset[key].append(ele[key])
    hf_dataset = Dataset.from_dict(dict_dataset)
    hf_dataset.push_to_hub("Dahoas/code-review-instruct-critique-revision")

if __name__ == "__main__":
    #inspect_output()
    upload_dataset()
