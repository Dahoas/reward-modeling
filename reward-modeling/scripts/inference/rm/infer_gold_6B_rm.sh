accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/static-hh \
--log_file gptj_rm_static_rm_on_full_static_test \
--model_name EleutherAI/gpt-j-6B \
--tokenizer_name EleutherAI/gpt-j-6B \
--split test \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/gptj-rm/good_ckpt/hf_ckpt.pt \
--order chosen rejected \
#--save_model