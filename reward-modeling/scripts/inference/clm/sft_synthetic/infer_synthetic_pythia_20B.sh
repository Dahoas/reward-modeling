accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/synthetic_eval \
--log_file pythia_synthetic_20B_synthetic_eval \
--model_name Dahoas/synthetic-gptneox-sft-static \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 4