deepspeed --num_gpus=8 finetune_rm.py --config_path ../configs/IHP/gptj-rm.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_j.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_j.json