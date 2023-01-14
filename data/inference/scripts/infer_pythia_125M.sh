accelerate launch --config_file accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file pythia_125M_inference \
--model_name Dahoas/pythia-125M-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b
