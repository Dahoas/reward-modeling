deepspeed --num_gpus=8 finetune_base.py --config_path ../configs/pythia/base_configs/6B.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_2.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_2.json