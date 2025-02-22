ray stop --force

set -x

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/deepseek-v3-9B
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/deepseek-v3-9B
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/deepseek-v3-9B

# 训练长度
max_prompt_length=2048
max_response_length=16384
# batch size && 训练epoch
train_batch_size=256
ppo_mini_batch_size=16
val_batch_size=6834
total_epochs=200
test_freq=5
save_freq=-1
# 算法相关的参数
actor_lr=1e-5
critic_lr=2e-6
lr_warmup_steps=10
kl_coef=0.0
use_last_response=False
use_ref_answer=False
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.0
upgo_loss_version=1
clip_ratio2=2.0
weight_decay=0.1
adv_estimator=grpo
kl_loss_weight=0.1
num_bon=1
bon_strategy=all
kl_penalty=low_var_kl
# tracking实验名
project_name='verl_example_math'
experiment_name='deepseek_v3_grpo'

recipe=tasks_scripts/recipes/h20/dsv3_9b_grpo.yaml

python3 tasks/main_ppo.py \
    recipe=${recipe} \
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
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=256 \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.model.path=${RM_MODEL_PATH} \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=False \
    reward_model.punish_no_answer=v0 \
    reward_model.add_int_verify=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    +actor_rollout_ref.rollout.complete_ratio=1.0 \
    +actor_rollout_ref.rollout.max_off_policy_steps=5 \
    actor_rollout_ref.actor.kl_loss_weight=${kl_loss_weight} \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    +actor_rollout_ref.rollout.use_vllm=True