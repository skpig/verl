set -x
ray stop --force

MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan1/sft/M8_680m_PT/alpha
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/M8_680m_RM

# local env
export OMP_NUM_THREADS=32
export HDFS_IO_THROW_EXCEPTION=1
export PYTHONPATH=$PYTHONPATH:/data03/home/liuxin.ai/seed_models:/data03/home/liuxin.ai/verl


python3 tasks/main_rm.py \
    data.train_files=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan/rm/data/train.parquet \
    data.val_files=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan/rm/data/val.parquet \
    data.train_batch_size=128 \
    data.micro_batch_size=128 \
    model.path=$MODEL_PATH \
    trainer.default_hdfs_dir=$default_hdfs_dir \
    trainer.project_name=alphaseed-rm \
    trainer.experiment_name=rm_test_m8_680m \
    trainer.total_epochs=1 \
    trainer.logger=['console','tracking'] \
    data.max_seq_length=8192 \
    data.max_token_len=8192 \
    data.truncation=left \
    model.enable_gradient_checkpointing=true \
    model.use_rmpad=True \
    model.use_dynamic_bsz=True \
    model.sp_size=2 \
    model.tp_size=1 \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.01 \
    optim.min_lr_ratio=0.1 \
    +model.override_config.attention_dropout=0. \
    +model.override_config.embd_pdrop=0. \
    +model.override_config.resid_pdrop=0. \
    +model.override_config._moe_implementation=fused \
    trainer.nnodes=1
