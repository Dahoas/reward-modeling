accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file synthetic_1B_rm_on_hh_test \
--model_name Dahoas/pythia-1B-static-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split test \
--batch_size 4 \
--rm_path /fsx/alex/ckpts/pythia/synthetic-1B-rm/hf_ckpt/hf_ckpt.pt