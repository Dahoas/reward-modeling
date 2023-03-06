accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_prompted_human_eval \
--log_file logs/pythia_prompted_125M_hh_eval \
--model_name EleutherAI/pythia-160m-deduped \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1