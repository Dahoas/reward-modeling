accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_prompted_human_eval \
--log_file logs/pythia_prompted_6B_hh_eval \
--model_name EleutherAI/pythia-6.9b-deduped \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1
