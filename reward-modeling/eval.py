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
import torch.nn.functional as F
import argparse

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


def compute_seq_kl(logprobs, ref_logprobs):
    # TODO(alex): modify to only compute on response tokens
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    log_ratio = logprobs - ref_logprobs
    ratio = torch.exp(log_ratio)
    approx_kl = torch.mean((ratio - 1) - log_ratio)#torch.mean(torch.sum((ratio - 1) - log_ratio, dim=1))
    return approx_kl


def compute_kl_on_dataset(m1, m2, tok, dataset, d1, d2):
    m1.eval().half()
    m2.eval().half()
    #d1, d2 = 0, 1
    m1.to(f"cuda:{d1}")
    m2.to(f"cuda:{d2}")

    tok.pad_token = tok.eos_token
    max_length = 1024
    def data_collator(data):
        prompts = [sample[0] for sample in data]
        responses = [sample[1] for sample in data]
        tok_prompts = tok(prompts, padding=False)["input_ids"]
        tok_responses = tok(responses, padding=False)["input_ids"]
        response_inds = [[len(tok_p), len(tok_p) + len(tok_r)] for tok_p, tok_r in zip(tok_prompts, tok_responses)]

        data = [sample[0] + sample[1] for sample in data]
        toks = tok(data, padding="longest", return_tensors="pt")

        return toks, response_inds

    batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    def select_logprobs(logprobs, inds):
        selected_logprobs = []
        for ind, logprob in zip(inds, logprobs):
            selected_logprobs.append(logprob[ind[0] : ind[1]])
        selected_logprobs = torch.nn.utils.rnn.pad_sequence(selected_logprobs, batch_first=True)
        return selected_logprobs

    kl_mean = 0
    kl_var = 0
    for i, batch in enumerate(dataloader):
        response_inds = batch[1]
        batch = batch[0]
        with torch.no_grad():
            logits = m1(batch["input_ids"].to(f"cuda:{d1}"), batch["attention_mask"].to(f"cuda:{d1}")).logits.cpu().type(torch.float32)
            ref_logits = m2(batch["input_ids"].to(f"cuda:{d2}"), batch["attention_mask"].to(f"cuda:{d2}")).logits.cpu().type(torch.float32)
        # Pick only distribution over actions taken
        logits = torch.gather(logits, 2, batch["input_ids"].unsqueeze(dim=-1)).squeeze(-1)
        ref_logits = torch.gather(ref_logits, 2, batch["input_ids"].unsqueeze(dim=-1)).squeeze(-1)
        logprobs = F.log_softmax(logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        logits = select_logprobs(logprobs, response_inds)
        ref_logits = select_logprobs(ref_logprobs, response_inds)
        new_kl = compute_seq_kl(logits, ref_logits)
        if i < 10 or new_kl < kl_mean + 3*kl_var**(1/2):
            kl_mean = (i*kl_mean + new_kl) / (i+1)
            kl_var = (i*kl_var + (kl_mean - new_kl)**2) / (i+1)
        if i % 40 == 0:
            print("Step {} of {}".format(i, len(dataloader)))
            print("KL mean: {} | KL var: {}".format(kl_mean, kl_var))
    print("KL mean: {} | KL std: {}".format(kl_mean, kl_var))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", default=0, type=int)
    parser.add_argument("--d2", default=1, type=int)
    parser.add_argument("--m1", default="EleutherAI/pythia-70m-deduped", type=str)
    parser.add_argument("--m2", default="EleutherAI/pythia-160m-deduped", type=str)
    args = parser.parse_args()
    #dataset = load_jsonl("6B_rm_on_synthetic_test.jsonl")
    #dataset = load_jsonl("1B_rm_inference_test.jsonl")
    #dataset = load_jsonl("gptj_rm_static_rm_on_full_static_test.jsonl")
    #order = ["instruct_3", "instruct_1", "20B", "6B"]
    #order = ["chosen", "rejected"]
    #compute_acc_from_reward_labeled_data(dataset, order=order)
    #label()
    #eval_rm_acc()
    #save_as_fp32()
    #gen_candidates()
    #m1 = AutoModelForCausalLM.from_pretrained("/mnt/nvme/home/alex/repos/rlhf/trlx/examples/hh/vanilla-pythia-6B-frozen-4-2e6-bs-4-BEST")
    #m1 = AutoModelForCausalLM.from_pretrained("Dahoas/pythia-6B-sft-response-full-static")
    #m2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    #m1 = AutoModelForCausalLM.from_pretrained("Dahoas/gptneox-response-full-static-sft")
    #m1 = AutoModelForCausalLM.from_pretrained("/mnt/nvme/home/alex/repos/rlhf/trlx/examples/hh/vanilla-gptneox-frozen-4-4e6")
    #m2 = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
    #m1 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")
    #m2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped")
    m1 = AutoModelForCausalLM.from_pretrained(args.m1)
    m2 = AutoModelForCausalLM.from_pretrained(args.m2)

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dataset = []
    for sample in load_dataset("Dahoas/static-hh")["train"]:
        dataset.append([sample["prompt"], sample["chosen"]])
    compute_kl_on_dataset(m1, m2, tok, dataset, args.d1, args.d2)
