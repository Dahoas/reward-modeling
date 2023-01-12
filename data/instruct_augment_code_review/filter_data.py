from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import re
import torch
from utils import extract_max_code_block


def filter_dataset(base_dataset):
    tok = AutoTokenizer.from_pretrained("gpt2")
    MAX_LENGTH = 2048
    CODE_BLOCK_THRESHOLD = 5

    new_dataset = {key: [] for key in base_dataset[0].keys()}
    cnt = 0
    for post in tqdm(base_dataset):
        cnt += 1
        body = post["body"]
        bl = len(tok(body)["input_ids"])
        
        new_answers = []
        for answer in post["answers"]:
            al = len(tok(answer["body"])["input_ids"])
            if bl + al <= MAX_LENGTH:
                max_code_block = extract_max_code_block(answer["body"])
                if max_code_block is not None and len(max_code_block.split(" ")) > CODE_BLOCK_THRESHOLD:
                    new_answers.append(answer)
        
        if len(new_answers) > 0:
            new_dataset["body"].append(body)
            new_dataset["answers"].append(new_answers)
            new_dataset["comments"].append(post["comments"])
            new_dataset["meta_data"].append(post["meta_data"])
            new_dataset["question_id"].append(post["question_id"])

    new_dataset = Dataset.from_dict(new_dataset)
    new_dataset.push_to_hub("Dahoas/2048_has_code_filtered_base_code_review")


def reformat_by_question(base_dataset):
    new_dataset = {"body": [], "comments": [], "answer": [], "meta_data": [], "question_id": []}
    for sample in new_dataset:
        for answer in sample["answers"]:
            new_dataset["body"].append(sample["body"])




if __name__ == "__main__":
    #base_dataset = load_dataset("Dahoas/base_code_review")["train"]
    #filter_dataset(base_dataset)
    base_dataset = load_dataset("Dahoas/2048_has_code_filtered_base_code_review")["train"]
    reformat_by_question(base_dataset)