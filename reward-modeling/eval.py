from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_prompts
import json
import torch
from reward_model import GPTRewardModel
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from logger import Logger


def load_jsonl(filename):
	data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			response = json.loads(line)
			data.append(response)
	return data


def eval_supervised():
    #temp_name = "gpt2"
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    #gptj = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:0")
    gptj = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to("cuda:0")
    gptj.model = "cuda:0"

    supervised_gptj = AutoModelForCausalLM.from_pretrained("ckpts/supervised_chosen_batched").to("cuda:1")
    supervised_gptj.model = "cuda:1"

    prompts = load_prompts("prompts_tiny.jsonl")

    def eval(model, tokenizer, dataset, file_path, gen_kwargs, batch_size=20):
        with open(file_path, "w") as f:
            batches = [dataset[batch_size*i : batch_size*(i+1)] for i in range((len(dataset) + batch_size - 1) // batch_size)]
            for batch in batches:
                inputs = tokenizer(batch, padding="longest",return_tensors="pt").to(model.device)  # Asserts existence of model.device
                response = model.generate(**inputs, **gen_kwargs)[:, inputs["input_ids"].shape[1]:]
                decoded_responses = tokenizer.batch_decode(response)
                for prompt, text in zip(batch, decoded_responses):
                    json.dump({"prompt": prompt, "response": text}, f)
                    f.write("\n")

    gen_kwargs = {"do_sample":True, "top_k":50, "max_length":600, "top_p":0.95, "temperature":1.9}
    eval(gptj, tokenizer, prompts, "gptj_evals.jsonl", gen_kwargs)
    eval(supervised_gptj, tokenizer, prompts, "supervised_gptj_evals.jsonl", gen_kwargs)


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer('<|startoftext|>' + chosen + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            rejected_encodings_dict = tokenizer('<|startoftext|>' + rejected + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]

def data_collator(data):
    return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
            'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}

def eval_rm():
    data = []
    dataset_name = "single_context_pairwise_test"
    with open(dataset_name + ".jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line)
    print("Len data: ", len(data))

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 1024

    dataset = PairwiseDataset(data, tokenizer, max_length=max_length)
    batch_size = 24
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    model = GPTRewardModel("EleutherAI/gpt-neo-2.7B")
    model.load_state_dict(torch.load("ckpts/single_context_pairwise/gpt-neo2.7B-one-epoch.pt"))
    model.half() # Converts to fp16 for faster inference
    model.eval()
    model.cuda()

    def gen(inputs):
        input_ids = inputs["input_ids"].to("cuda")
        #print(torch.cuda.memory_summary())
        with torch.no_grad():
            output = model(input_ids)
        # Correct reward score is last element - unless there are issues with padding?
        rewards = output[:, -1]
        return rewards

    cnt = 0
    for i, batch in tqdm(enumerate(dataloader)):
        #if i > 1000:
        #    break
        rewards = gen(batch)
        chosen_rewards = rewards[:rewards.shape[0] // 2]
        rejected_rewards = rewards[rewards.shape[0] // 2:]
        #torch.cuda.empty_cache()
        cnt += torch.sum(chosen_rewards > rejected_rewards).item()
        #print(torch.cuda.memory_summary())
    print(cnt / len(data))


def label():
    data = []
    dataset_name = "single_context_pairwise"
    with open(dataset_name + ".jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line)
    print("Len data: ", len(data))

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 1024

    dataset = PairwiseDataset(data, tokenizer, max_length=max_length)
    batch_size = 12
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    model = GPTRewardModel("EleutherAI/gpt-j-6B")
    model.load_state_dict(torch.load("ckpts/single_context_pairwise/gpt-j.pt"))
    model.half() # Converts to fp16 for faster inference
    model.eval()
    model.cuda()

    def gen(inputs):
        input_ids = inputs["input_ids"].to("cuda")
        #print(torch.cuda.memory_summary())
        with torch.no_grad():
            output = model(input_ids)
        # Correct reward score is last element - unless there are issues with padding?
        rewards = output[:, -1]
        return rewards

    cnt = 0
    chosen_rewards = []
    rejected_rewards = []
    with open("rm_labeled_single_context_pairwise.jsonl", "w") as f:
        for i, batch in tqdm(enumerate(dataloader)):
            #if i > 1000:
            #    break
            rewards = gen(batch)
            chosen_rewards += rewards[:rewards.shape[0] // 2].tolist()
            rejected_rewards += rewards[rewards.shape[0] // 2:].tolist()

        for data_element, chosen_reward, rejected_reward in zip(data, chosen_rewards, rejected_rewards):
            data_element["chosen_reward"] = chosen_reward
            data_element["rejected_reward"] = rejected_reward
            json.dump(data_element, f)
            f.write("\n")

        chosen_rewards = []
        rejected_rewards = []



def save_as_fp32():
    model = GPTRewardModel("EleutherAI/gpt-j-6B")
    model = load_state_dict_from_zero_checkpoint(model, "ckpts/single_context_pairwise/gpt-j/checkpoint-66525")
    torch.save(model.state_dict(), "ckpts/single_context_pairwise/gpt-j.pt")

def gen_candidates():
    Logger.init("gptj_completions")

    model = AutoModelForCausalLM.from_pretrained("ckpts/supervised_chosen_batched")
    model = model.to("cuda")
    prompts = load_jsonl("extracted_synthetic_alignment_prompts.jsonl")
    prompts = [prompt["prompt"] for prompt in prompts]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length = 1024

    def data_collator(data):
        tok_prompts = tokenizer(data, padding="longest", return_tensors="pt")
        tok_prompts["input_ids"] = tok_prompts["input_ids"].to("cuda")
        tok_prompts["attention_mask"] = tok_prompts["attention_mask"].to("cuda")
        return tok_prompts, data

    batch_size = 4
    dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, collate_fn=data_collator)

    responses = []
    for inputs in tqdm(dataloader):
        input_texts = inputs[1]
        inputs = inputs[0]
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True)[:, inputs["input_ids"].shape[1]:]
        text_outputs = tokenizer.batch_decode(outputs)
        responses += text_outputs
        for input_text, output_text in zip(input_texts, text_outputs):
            output_text = output_text.split("<|endoftext|>")[0]
            Logger.log([{"prompt": input_text, "response": output_text}])



if __name__ == "__main__":
    #label()
    #eval_rm()
    #save_as_fp32()
    gen_candidates()
