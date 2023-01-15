accelerate launch --config_file accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file pythia_1B_inference_train \
--model_name Dahoas/pythia-1B-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b
