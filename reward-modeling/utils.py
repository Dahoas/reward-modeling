import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, AutoModel
from torch import nn
import functools
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


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


def make_rm(model_name):
    config = AutoConfig.from_pretrained("gpt2")
    config.num_labels = 1
    reward_model = AutoModelForSequenceClassification.from_config(config)
    return reward_model


def upload_model():
    model_path = "../ckpts/gpt2-sft/"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.push_to_hub(repo_url="https://huggingface.co/Dahoas/gpt2-sft-single-context")


def convert_deepspeed_checkpoint(is_rm=True):
    model_name = None
    model_path = None
    if is_rm:
        model = make_rm(model_name)
    else:
        AutoModel.from_pretrained(model_name)
    fp32_model = load_state_dict_from_zero_checkpoint(model, 'results/checkpoint-134')


if __name__ == "__main__":
    upload_model()