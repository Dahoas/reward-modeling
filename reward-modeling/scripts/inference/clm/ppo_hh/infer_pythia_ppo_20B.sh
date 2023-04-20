accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/hh_human_eval \
--log_file logs/pythia_20B_ppo_hh_eval \
--model_name  /mnt/nvme/home/alex/repos/rlhf/trlx/20B-frozen-4-2e6-bs-4-mbs-1-nodes-4-BEST \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 1