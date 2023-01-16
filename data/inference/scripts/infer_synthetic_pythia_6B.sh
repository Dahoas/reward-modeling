accelerate launch --config_file accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-synthetic-hh \
--log_file pythia_synthetic_6B_inference_test \
--model_name Dahoas/pythia-synthetic-6B-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split test \
--batch_size 2