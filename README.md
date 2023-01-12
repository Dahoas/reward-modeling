# reward-modeling

This is a research repository for training and evaluating reward models. Code is also included to train supervised fine-tuned base models.

## Example

Running `bash scripts/train_rm/run_gptj.sh` will train a `gptj` reward model using train config `configs/rm_configs/gptj.yaml` by default. 

**Note**: To do eval on gptj please install [transformers](https://github.com/huggingface/transformers) repo from source.
