accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_human_eval \
--log_file logs/pythia_sft_125M_hh_eval \
--model_name Dahoas/pythia-125M-response-full-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1