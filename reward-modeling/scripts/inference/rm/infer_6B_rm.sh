accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file 6B_rm_inference_train \
--model_name EleutherAI/gpt-j-6B \
--tokenizer_name EleutherAI/gpt-j-6B \
--split train \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/gptj-rm/good_ckpt/hf_ckpt.pt