import openai
from logger import Logger
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import time
from datasets import load_dataset


def query(prompts, max_tokens, temperature=1.0):
    responses = openai.Completion.create(engine='text-davinci-003', prompt=prompts, max_tokens=max_tokens, temperature=temperature)["choices"]
    responses = [response["text"] for response in responses]
    return responses

# Assumes prompts is a list of dicts, each of which contains a "prompt" key
def run_prompts(prompts, batch_size, max_tokens, temperature):
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    batched_prompts = [prompts[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    
    for batch in tqdm(batched_prompts):
        batch_prompts = [sample["prompt"] for sample in batch]
        try:
            responses = query(batch_prompts, max_tokens, temperature)
            for sample, response in zip(batch, responses):
                sample["response"] = response
            Logger.log(batch)
        except openai.error.RateLimitError:
            print("RATELIMIT ERROR")
            time.sleep(30)
        except openai.error.ServiceUnavailableError:
            print("SERVICE UNABAILABLE")
            time.sleep(15)
        except openai.error.Timeout:
            print("TIMEOUT")
            time.sleep(15)
        except:
            print("SOME OTHER EXCEPTION")
            time.sleep(30)
        time.sleep(15)  # Sleep to prevent rate limiting

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset")
    parser.add_argument("--log_file")
    parser.add_argument("--oai_key")
    args = parser.parse_args()

    Logger.init(args.log_file)
    openai.api_key = args.oai_key

    prompt_dataset = load_dataset(args.prompt_dataset)["train"]
    prompt_dataset = [{key: sample[key] for key in sample} for sample in prompt_dataset]
    prompt_dataset = prompt_dataset[21000:]
    run_prompts(prompt_dataset, 10, 1024+60, 1.4)
