accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_prompted_baseline_prompts \
--log_file pythia_6B_prompted_synthetic_eval \
--model_name Dahoas/pythia-6B-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2