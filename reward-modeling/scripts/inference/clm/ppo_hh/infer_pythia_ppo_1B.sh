accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_eval \
--log_file pythia_1B_ppo_hh_eval \
--model_name reciprocate/ppo_hh_pythia-1B \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2
