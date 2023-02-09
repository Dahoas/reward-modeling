from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import deepspeed
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
from rm_datasets import PairwiseDataset, PairwiseEvalDataset, pairwise_data_collator
from datasets import load_dataset
from utils import make_rm

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



# Launch with: deepspeed --num_gpus 1 eval.py
def eval_rm_acc(use_deepspeed=False):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    #model = AutoModelForSequenceClassification.from_pretrained("Dahoas/gptj-rm-static")
    model_name = "EleutherAI/gpt-neo-1.3B"
    model = make_rm(model_name, "causal")
    model.load_state_dict(torch.load("../ckpts/gptneo-rm/hf_ckpt.pt"))
    model.config.pad_token_id = model.config.eos_token_id
    world_size = 1
    if use_deepspeed:
        model = deepspeed.init_inference(model,
                                                mp_size=world_size,
                                                dtype=torch.float,
                                                replace_method='auto',
                            replace_with_kernel_inject=True)
    else:
        model.eval()
        model.half()
        model.cuda()

    data = load_dataset("Dahoas/rm-static")
    max_length = 1024
    eval_dataset = PairwiseDataset(data["test"], tokenizer, max_length=max_length)
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=pairwise_data_collator)

    cnt = 0
    for i, batch in tqdm(enumerate(dataloader)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        rewards = model(input_ids, attention_mask=attention_mask)
        chosen_rewards = rewards[:rewards.shape[0] // 2]
        rejected_rewards = rewards[rewards.shape[0] // 2:]
        cnt += torch.sum(chosen_rewards > rejected_rewards).item()
    print(cnt / len(data["test"]))




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


def gather_group(dataset, Id):
    group = {}
    ids = []
    for i, s in enumerate(dataset):
            if s["id"] == Id:
                if group.get(s["type"]) is not None:
                    ValueError("Conflicting sample types for sample {}".format(sample["id"]))
                group[s["type"]] = s
                ids.append(i)
    return ids, group

def compute_acc_from_reward_labeled_data(dataset, order=["chosen", "rejected"]):
    num_classes = len(order)
    accs = {}
    cnt = 0
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            accs["{}-{}".format(order[i], order[j])] = {"acc": 0, "cnt": 0}

    lonely = []
    while len(dataset) > 0:
        sample = dataset.pop(0)
        ids, group = gather_group(dataset, sample["id"])
        group[sample["type"]] = sample

        if len(group) != num_classes:
            print("Sample {} is imbalanced".format(sample["id"]))
            lonely.append(sample)
            continue
        
        # Remove sieved samples from dataset
        for ind in ids[::-1]:
            dataset.pop(ind)

        for i in range(num_classes):
            for j in range(i+1, num_classes):
                name = "{}-{}".format(order[i], order[j])
                accs[name]["acc"] += int(group[order[i]]["reward"] > group[order[j]]["reward"])
                accs[name]["cnt"] += 1
                cnt += 1

        if cnt % 1000 == 0:
            print("Pair {}...".format(cnt))
            print(group)

    acc = 0
    for key in accs:
        acc += accs[key]["acc"]
        accs[key]["acc"] = accs[key]["acc"] / accs[key]["cnt"]
    acc = acc / cnt
    print(json.dumps(accs, indent=2))
    print("Dataset acc: {}".format(acc))

if __name__ == "__main__":
    #dataset = load_jsonl("6B_rm_on_synthetic_test.jsonl")
    #dataset = load_jsonl("1B_rm_inference_test.jsonl")
    dataset = load_jsonl("augmented_synthetic_6B_rm_on_hh_test.jsonl")
    #order = ["instruct_3", "instruct_1", "20B", "6B"]
    order = ["chosen", "rejected"]
    compute_acc_from_reward_labeled_data(dataset, order=order)
    #label()
    #eval_rm_acc()
    #save_as_fp32()
    #gen_candidates()
