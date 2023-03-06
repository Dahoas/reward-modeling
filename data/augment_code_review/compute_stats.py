from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def compute_stats(base_dataset):
    base_dataset = base_dataset.shuffle()
    tok = AutoTokenizer.from_pretrained("gpt2")

    stats = {"avg_body_len": 0, "avg_answer_len": 0, "max_body_len": 0, "max_answer_len": 0, "avg_num_answers": 0}
    cnt = 0
    for post in tqdm(base_dataset):
        if cnt > 5000:
            break
        cnt += 1
        body = post["body"]
        l = len(tok(body)["input_ids"])
        stats["avg_body_len"] += l
        stats["max_body_len"] = max(l, stats["max_body_len"])
        
        for answer in post["answers"]:
            l = len(tok(answer["body"])["input_ids"])
            stats["avg_answer_len"] += l
            stats["max_answer_len"] = max(l, stats["max_answer_len"])
        
        stats["avg_num_answers"] += len(post["answers"])

    stats["avg_body_len"] = stats["avg_body_len"] / cnt
    stats["avg_answer_len"] = stats["avg_answer_len"] / stats["avg_num_answers"]
    stats["avg_num_answers"] = stats["avg_num_answers"] / cnt
        
    print(stats)

if __name__ == "__main__":
    base_dataset = load_dataset("Dahoas/2048_has_code_filtered_base_code_review")["train"]
    compute_stats(base_dataset)