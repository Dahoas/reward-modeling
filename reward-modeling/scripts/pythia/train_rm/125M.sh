deepspeed --num_gpus=8 finetune_rm.py --config_path ../configs/pythia/rm_configs/125M.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_2.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_2.json