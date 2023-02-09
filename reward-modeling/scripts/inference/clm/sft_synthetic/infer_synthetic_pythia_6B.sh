accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_synthetic_eval \
--log_file pythia_synthetic_6B_hh_synthetic_eval \
--model_name Dahoas/pythia-synthetic-6B-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2