accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file 1B_rm_inference_test \
--model_name EleutherAI/gpt-neo-1.3B \
--tokenizer_name EleutherAI/gpt-neo-1.3B \
--split test \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/gptneo-rm/hf_ckpt/hf_ckpt.pt