set -x
ray stop --force

NUM_STEPS="${NUM_STEPS:-240}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# local env
export OMP_NUM_THREADS=32
export HDFS_IO_THROW_EXCEPTION=1
export XPERF_DUMP_NAN=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RAY_prestart_worker_first_driver=0
export PYTHONPATH=$PYTHONPATH:/data03/home/liuxin.ai/seed_models:/data03/home/liuxin.ai/verl

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying/projects/alphaseed/datasets/opensource/deepscaler_train_40k.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying/projects/alphaseed/datasets/opensource/deepscaler_format_aime_eval.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/p6_400m_test

# 训练长度
max_prompt_length=1024 # 16384
max_response_length=8192 # 16384
# batch size && 训练epoch
train_batch_size=128
ppo_mini_batch_size=128
total_epochs=100
test_freq=5
save_freq=-1
# 算法相关的参数
actor_lr=1e-6
lr_warmup_steps_ratio=0.0003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0
kl_loss_weight=0.0
entropy_coeff=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
adv_estimator=grpo
kl_penalty=low_var_kl
num_bon=8
bon_strategy=all
scale_pg_by_kl=False

# 工程参数
actor_sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=24576
gen_tp=1

# tracking实验名
project_name='verl_example_math_lx'
experiment_name='p6_400m_math-v1'
export PYTHONPATH=$PYTHONPATH:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton
python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.truncation='left' \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    actor_rollout_ref.actor.scale_pg_by_kl=${scale_pg_by_kl} \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.micro_batch_size=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=True \
    +actor_rollout_ref.rollout.enable_paged_attention=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    +actor_rollout_ref.rollout.complete_ratio=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.actor.act_offload=True \
    reward_model.enable=False \
    reward_model.mean=0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.punish_format=False \
    reward_model.format_punish_score=-0.01 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.save_cases_to_hdfs=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps=disable \
    trainer.offload_train_memory=True \
    trainer.total_steps=${NUM_STEPS} \
    server_client.role="server" \
    2>&1 | tee log.txt
