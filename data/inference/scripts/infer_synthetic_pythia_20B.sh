accelerate launch --config_file accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-synthetic-hh \
--log_file pythia_synthetic_20B_inference_train \
--model_name Dahoas/synthetic-gptneox-sft-static \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 8