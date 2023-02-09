accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_eval \
--log_file pythia_20B_sft_hh_eval \
--model_name Dahoas/gptneox-sft-static \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2
