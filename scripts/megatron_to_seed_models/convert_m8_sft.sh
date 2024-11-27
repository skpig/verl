hdfs_path=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan/sft/M8_2.5B/M8_2.5B/25B_MoE_SFT29_32k_bsz6_lr2e5__tp4
checkpoint_path=global_step_1036

cd /opt/tiger/seed_models && pip3 install -e . --user && \


python3 -m seed_models.commands.convert \
    --omnistore_ckpt_path=${hdfs_path}/checkpoints/${checkpoint_path} \
    --hf_path=/opt/tiger/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf \
    --model_type=M8 \
    --cruise_config_path=${hdfs_path}/cruise_cli.yaml \
    --save_safetensors \
    --output_path=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models \
    --auto_model=ForCausalLM