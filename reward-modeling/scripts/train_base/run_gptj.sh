deepspeed --num_gpus=8 finetune_base.py --config_path ../configs/base_configs/gptj.yaml \
--ds_config_path ../configs/ds_configs/ds_config_gpt_j.json \
--deepspeed ../configs/ds_configs/ds_config_gpt_j.json