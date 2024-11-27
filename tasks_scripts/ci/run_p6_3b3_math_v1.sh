set -x

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf_new
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_3b3_0812_sftv27_stage2_fix_order_aux_32k_v2_hf
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/test/p6_3b3

# 训练长度
max_prompt_length=1024 # 16384
max_response_length=2048 # 16384
# batch size && 训练epoch
train_batch_size=1024
val_batch_size=5000
ppo_mini_batch_size=256
ppo_micro_batch_size=256
total_epochs=100
test_freq=5
save_freq=-1
# 算法相关的参数
actor_lr=1e-6
critic_lr=1e-5
lr_warmup_steps_ratio=0.0003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.001
entropy_coeff=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
# tracking实验名
project_name='verl_example_math_ci'
experiment_name='p6moe_3b3_math-v1'

python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    +data.chat_template=seed \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.rollout.micro_batch_size=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${ppo_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=512 \
    reward_model.mean=0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.reward_0_for_overlong_rsp=True \
    reward_model.punish_no_answer=v0 \
    +reward_model.use_rmpad=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl