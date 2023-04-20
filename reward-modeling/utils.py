import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from torch import nn
import functools
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import os
from reward_models import RewardModel
import torch
import json


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line)
    return data


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)


def hf_get_causal_hidden_layers(model: nn.Module):
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "transformer.layers",
    )
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_causal_hidden_layers(model)
    num_layers_unfrozen = int(len(hidden_layers) * num_layers_unfrozen) if type(num_layers_unfrozen) is float else num_layers_unfrozen
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def make_rm(model_name, type_t, tok_path, save_model):
    if type_t == "classification":
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        reward_model = AutoModelForSequenceClassification.from_config(config)
    elif type_t == "causal":
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        reward_model = RewardModel(model_name, tokenizer(tokenizer.eos_token)["input_ids"][0], save_model)
    else:
        raise ValueError("Unsupported reward model type {}".format(type_t))
    return reward_model


def upload_model():
    model_path = "../ckpts/gpt2-sft/"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.push_to_hub(repo_url="https://huggingface.co/Dahoas/gpt2-sft-single-context")


def convert_deepspeed_checkpoint(is_rm=True):
    model_name = "EleutherAI/gpt-j-6B" # "Dahoas/pythia-6B-static-sft"
    tok_name = "EleutherAI/gpt-j-6B" #"EleutherAI/gpt-neox-20b"
    model_path = "/mnt/nvme/home/alex/repos/rlhf/ckpts/gptj-rm-IHP"
    model_ckpt = "checkpoint-10633"
    type_t = "causal"
    if is_rm:
        model = make_rm(model_name, type_t, tok_name, True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    fp32_model = load_state_dict_from_zero_checkpoint(model, os.path.join(model_path, model_ckpt))
    if type_t == "causal" and is_rm:
        if not os.path.exists(os.path.join(model_path, "hf_ckpt")):
            os.mkdir(os.path.join(model_path, "hf_ckpt"))
        torch.save(model.state_dict(), os.path.join(model_path, "hf_ckpt/hf_ckpt.pt"))
    else:
        fp32_model.save_pretrained(os.path.join(model_path, "hf_ckpt"))

def split_ckpt(num_chunks):
    ckpt_path = "/fsx/alex/ckpts/gptneox-sft/hf_ckpt"
    print("Splitting {} ...".format(ckpt_path))
    sd = torch.load(os.path.join(ckpt_path, "hf_ckpt.pt"))
    keys = list(sd.keys())
    chunk_size = len(keys) // num_chunks
    num_chunks = (len(keys) + chunk_size - 1) // chunk_size
    key_batches = [keys[chunk_size*i : chunk_size * (i+1)] for i in range(num_chunks)]
    index = {}
    for i, key_batch in enumerate(key_batches):
        sub_dict_name = "hf_ckpt_{}.pt".format(i)
        print("Saving {} ...".format(sub_dict_name))
        sub_dict = {key: sd[key] for key in key_batch}
        for key in key_batch:
            index[key] = sub_dict_name
        torch.save(sub_dict, os.path.join(ckpt_path, sub_dict_name))
        del sub_dict
    with open(os.path.join(ckpt_path, "index.json"), "w") as f:
        json.dump(index, f, indent=4)
    print("Done!")

def hf_upload(make_repo=True):
    import os
    from huggingface_hub import HfApi, create_repo
    converted_ckpt = "/mnt/nvme/home/alex/repos/rlhf/ckpts/gptj-rm-IHP/hf_ckpt"
    repo_name = "Dahoas/gptj-rm-IHP"
    if make_repo:
        create_repo(repo_name, repo_type="model", private=False)

    files = os.listdir(converted_ckpt)
    api = HfApi()
    print(f"to upload: {files}")
    for file in files:
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=os.path.join(converted_ckpt, file),
            path_in_repo=file,
            repo_id=repo_name,
            repo_type="model",
            commit_message=f"Upload {file}",
        )
        print(f"Successfully uploaded {file} !")

if __name__ == "__main__":
    convert_deepspeed_checkpoint(is_rm=True)
    #split_ckpt(46)
    hf_upload(make_repo=True)