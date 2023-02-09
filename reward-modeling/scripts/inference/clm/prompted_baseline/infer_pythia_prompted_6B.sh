accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_prompted_baseline_prompts \
--log_file pythia_6B_prompted_hh_eval \
--model_name EleutherAI/pythia-6.9b-deduped \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1
