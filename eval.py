from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_prompts
import json

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