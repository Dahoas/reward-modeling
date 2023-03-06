accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_human_eval \
--log_file logs/pythia_sft_20B_hh_eval \
--model_name Dahoas/gptneox-response-full-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1
