# reward-modeling

This is a research repository for training and evaluating reward models. Code is also included to train supervised fine-tuned base models.

## Training Example

Running `bash scripts/train_rm/run_gptj.sh` will train a `gptj` reward model using train config `configs/rm_configs/gptj.yaml` by default. 

**Note**: To do eval on gptj please install [transformers](https://github.com/huggingface/transformers) repo from source.

## Loading Models

Loading models is a bit convoluted so I attach an example here. The reward models are not implemented as HF models and so cannot simply be loaded via a `.from_pretrained(MODEL_NAME)` call.

1. Get model weights from hf: `wget https://huggingface.co/Dahoas/pythia-6b-rm-synthetic/blob/main/hf_ckpt.pt`
2. Instantiate a model with the same base class as the ckpt: `model = AutoModelFromCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped")`
3. ```python
      import torch
      from utils import make_rm
      # save_model is used to determine whether a reference to the base model is saved in the RM wrapper (this is necessary to use HF's Activation Checkpointing code)
      save_model = False
      rm = make_rm("EleutherAI/gpt-j-6B", "causal", "EleutherAI/gpt-neox-20b", save_model)
      rm.load_state_dict(torch.load(PATH_TO_CKPT), strict=True)
      ```
