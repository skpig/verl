set -x

PYTHONPATH_ORIGIN=::/opt/tiger/arnold_toolbox:/opt/tiger/rh2:/opt/tiger/rh2:/opt/tiger/pyutil:/python:/python/lib/py4j-0.10.9-src.zip:/opt/tiger/arnold_toolbox:/opt/tiger/alpha-seed:/opt/tiger/rh2:/opt/tiger/pyutil
PYTHONPATH_THIS=/opt/tiger/mono_rl:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/nccl
export PYTHONPATH=$PYTHONPATH_ORIGIN:$PYTHONPATH_THIS

export UCX_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
NUM_STEPS="${NUM_STEPS:-10000}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
N_GPUS_PER_NODE_STREAMING=2

# ckpt和路径
# SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf
#SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/Qwen2.5-1.5B  # 从ds蒸馏的模型，能吐很长，chat tpl用raw
#SFT_MODEL_CHAT_TPL=seed
SFT_MODEL_CHAT_TPL=raw
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/M8_680m_RM/checkpoints/global_step_308/huggingface
#TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/yueyu/data/rlhf/d62_rename_datasource.parquet  # 这个会生成很长！用来测试
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans_top_100.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/x.lixiang/test/m8_680m_ppo_fsdp_async_nightly_ci

# 训练长度
max_prompt_length=2048
max_response_length=2048
# batch size && 训练epoch
train_batch_size=20  # 这个除以ppo_mini_batch_size得到更新optimizer的次数，gbsz越多，rollout gen的时间越长
ppo_mini_batch_size=20  # 会平分给每个dp
total_epochs=100
test_freq=-1  # 暂时跳过validate
save_freq=-1

# 算法相关的参数
actor_lr=1e-6
critic_lr=2e-6
lr_warmup_steps_ratio=0.0003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0
kl_loss_weight=0.0
entropy_coeff=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=1.0
adv_estimator=gae
kl_penalty=low_var_kl
num_bon=1
bon_strategy=all
scale_pg_by_kl=False

# 工程参数
actor_sp_size=1
critic_sp_size=1
gen_tp=2
use_dynamic_bsz=True
ppo_max_token_len=24576
# tracking实验名
project_name='alphaseed_timeline_lixiang'
experiment_name='p6_400m_math-v1_ppo_fsdp_async'

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
    +data.chat_template=${SFT_MODEL_CHAT_TPL} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${ppo_max_token_len} \
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
    +actor_rollout_ref.rollout.enable_paged_attention=True \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.actor.act_offload=False \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len=${ppo_max_token_len} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    critic.act_offload=True \
    reward_model.enable=False \
    reward_model.mean=0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.punish_no_answer=v0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=1 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.save_cases_to_hdfs=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps=disable \
    actor_rollout_ref.rollout.schedule_strategy="fifo" \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    trainer.offload_train_memory=True \
    trainer.total_steps=${NUM_STEPS} \
    streaming_validator.nnodes=0 \
    streaming_validator.n_gpus_per_node=${N_GPUS_PER_NODE_STREAMING} \
    +actor_rollout_ref.rollout.complete_ratio=1 \
    +actor_rollout_ref.rollout.max_off_policy_steps=5 \
    actor_rollout_ref.rollout.rollout_pool.warmup_step=1 \
    actor_rollout_ref.rollout.mode=server \
    elastic.enable=False \
    streaming_rollout.elastic.enable=False \
    streaming_rollout.nnodes=0 \
    streaming_rollout.n_gpus_per_node=${N_GPUS_PER_NODE_STREAMING}
