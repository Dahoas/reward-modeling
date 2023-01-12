import openai
from logger import Logger
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import time
from datasets import load_dataset
from utils import extract_max_code_block, filter_queried


def query(prompt_batch, max_tokens):
    prompts = []
    for sample in prompt_batch:
        body = sample["body"]
        answer = sample["answer"]["body"]
        prompt = "Question: {} \n\n Answer: {} \n\n This is a question and answer from a forum where users review and improve the code of other users. Please output the original code, a summary of the critique, and the revised code using the format ORIGINAL: [write original code here] CRITIQUE: [write critique here] REVISED: [write revision code here]. \n\n".format(body, answer)
        prompts.append(prompt)

    responses = openai.Completion.create(engine='text-davinci-003', prompt=prompts, max_tokens=max_tokens, temperature=0.1)["choices"]
    for prompt, response, sample in zip(prompts, responses, prompt_batch):
        text = response["text"]
        sample["prompt"] = prompt
        sample["response"] = text
        Logger.log([sample])
    responses = [response["text"] for response in responses]
    return responses


def augment_code_review():
    code_review_dataset = load_dataset("Dahoas/2048_has_code_filtered_base_code_review")["train"]
    reformatted_dataset = []
    for sample in code_review_dataset:
        for answer in sample["answers"]:
            reformatted_dataset.append({"body": sample["body"], "answer": answer, "comments": sample["comments"], "meta_data": sample["meta_data"], "question_id": sample["question_id"]})
    length = len(reformatted_dataset)
    code_review_dataset = filter_queried(reformatted_dataset)
    new_length = len(code_review_dataset)
    print("Old len: {}, New len: {}".format(length, new_length))

    prompts_per_query = 10
    batched_prompts = [code_review_dataset[i*prompts_per_query : (i+1)*prompts_per_query] for i in range((len(code_review_dataset) + prompts_per_query - 1) // prompts_per_query)]

    for prompt_batch in tqdm(batched_prompts):
        try:
            query(prompt_batch, 2048)
        except openai.error.RateLimitError:
            print("RATELIMIT ERROR")
            time.sleep(15)
        except openai.error.ServiceUnavailableError:
            print("SERVICE UNABAILABLE")
            time.sleep(15)
        except openai.error.Timeout:
            print("TIMEOUT")
            time.sleep(15)
        except:
            print("SOME OTHER EXCEPTION")
            time.sleep(30)
        time.sleep(10)  # Sleep to prevent rate limiting

def test():
    code_review_dataset = load_dataset("Dahoas/2048_has_code_filtered_base_code_review")["train"]
    sample = code_review_dataset[25002]
    body = sample["body"]
    answer = sample["answers"][0]["body"]

    prompt = "Question: {} \n\n Answer: {} \n\n This is a question and answer from a forum where users review and improve the code of other users. Please output the original code, a summary of the critique, and the revised code using the format ORIGINAL: [write original code here] CRITIQUE: [write critique here] REVISED: [write revision code here]. \n\n".format(body, answer)
    query([prompt], 2048)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file")
    parser.add_argument("--oai_key")
    args = parser.parse_args()

    Logger.init(args.log_file)
    openai.api_key = args.oai_key

    query_instruct()
