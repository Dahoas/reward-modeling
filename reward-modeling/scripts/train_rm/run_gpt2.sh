deepspeed --num_gpus=4 finetune_base.py --config_path ../configs/base_configs/gpt2.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_2.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_2.json