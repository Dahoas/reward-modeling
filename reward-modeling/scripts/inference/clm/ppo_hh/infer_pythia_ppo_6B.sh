accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_human_eval \
--log_file logs/pythia_6B_ppo_hh_eval \
--model_name /mnt/nvme/home/alex/repos/rlhf/trlx/examples/hh/pythia-6B-frozen-4-8e6 \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1
