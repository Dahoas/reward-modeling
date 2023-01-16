accelerate launch --config_file accelerate_configs/default.yaml \
inference.py \
--prompt_dataset Dahoas/rm-static \
--log_file pythia_20B_inference_test \
--model_name Dahoas/gptneox-sft-static \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split test \
--batch_size 1
