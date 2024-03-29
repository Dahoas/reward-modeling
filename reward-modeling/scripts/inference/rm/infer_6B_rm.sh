accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/static-hh \
--log_file gptj_full_static_rm_on_full_static_train \
--model_name EleutherAI/gpt-j-6B \
--tokenizer_name EleutherAI/gpt-j-6B \
--split train \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/gptj-6B-response-full-static-rm/hf_ckpt/hf_ckpt.pt \
--order chosen rejected \
--save_model True