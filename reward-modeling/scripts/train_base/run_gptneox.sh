deepspeed --num_gpus=8 finetune_base.py --config_path ../configs/base_configs/gptneox.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_neox.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_neox.json